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
def _base_environ(self, **request):
    """
        The base environment for a request.
        """
    return {'HTTP_COOKIE': '; '.join(sorted(('%s=%s' % (morsel.key, morsel.coded_value) for morsel in self.cookies.values()))), 'PATH_INFO': '/', 'REMOTE_ADDR': '127.0.0.1', 'REQUEST_METHOD': 'GET', 'SCRIPT_NAME': '', 'SERVER_NAME': 'testserver', 'SERVER_PORT': '80', 'SERVER_PROTOCOL': 'HTTP/1.1', 'wsgi.version': (1, 0), 'wsgi.url_scheme': 'http', 'wsgi.input': FakePayload(b''), 'wsgi.errors': self.errors, 'wsgi.multiprocess': True, 'wsgi.multithread': False, 'wsgi.run_once': False, **self.defaults, **request}