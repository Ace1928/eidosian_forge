import json
import mimetypes
import os
import sys
from copy import copy
from functools import partial
from http import HTTPStatus
from importlib import import_module
from io import BytesIO, IOBase
from urllib.parse import unquote_to_bytes, urljoin, urlparse, urlsplit
from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.core.handlers.base import BaseHandler
from django.core.handlers.wsgi import LimitedStream, WSGIRequest
from django.core.serializers.json import DjangoJSONEncoder
from django.core.signals import got_request_exception, request_finished, request_started
from django.db import close_old_connections
from django.http import HttpHeaders, HttpRequest, QueryDict, SimpleCookie
from django.test import signals
from django.test.utils import ContextList
from django.urls import resolve
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
from django.utils.http import urlencode
from django.utils.itercompat import is_iterable
from django.utils.regex_helper import _lazy_re_compile
class AsyncClientHandler(BaseHandler):
    """An async version of ClientHandler."""

    def __init__(self, enforce_csrf_checks=True, *args, **kwargs):
        self.enforce_csrf_checks = enforce_csrf_checks
        super().__init__(*args, **kwargs)

    async def __call__(self, scope):
        if self._middleware_chain is None:
            self.load_middleware(is_async=True)
        if '_body_file' in scope:
            body_file = scope.pop('_body_file')
        else:
            body_file = FakePayload('')
        request_started.disconnect(close_old_connections)
        await request_started.asend(sender=self.__class__, scope=scope)
        request_started.connect(close_old_connections)
        request = ASGIRequest(scope, LimitedStream(body_file, len(body_file)))
        request._dont_enforce_csrf_checks = not self.enforce_csrf_checks
        response = await self.get_response_async(request)
        conditional_content_removal(request, response)
        response.asgi_request = request
        if response.streaming:
            if response.is_async:
                response.streaming_content = aclosing_iterator_wrapper(response.streaming_content, response.close)
            else:
                response.streaming_content = closing_iterator_wrapper(response.streaming_content, response.close)
        else:
            request_finished.disconnect(close_old_connections)
            await sync_to_async(response.close, thread_sensitive=False)()
            request_finished.connect(close_old_connections)
        return response