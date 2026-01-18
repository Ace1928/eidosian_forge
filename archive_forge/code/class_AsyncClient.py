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
class AsyncClient(ClientMixin, AsyncRequestFactory):
    """
    An async version of Client that creates ASGIRequests and calls through an
    async request path.

    Does not currently support "follow" on its methods.
    """

    def __init__(self, enforce_csrf_checks=False, raise_request_exception=True, *, headers=None, **defaults):
        super().__init__(headers=headers, **defaults)
        self.handler = AsyncClientHandler(enforce_csrf_checks)
        self.raise_request_exception = raise_request_exception
        self.exc_info = None
        self.extra = None
        self.headers = None

    async def request(self, **request):
        """
        Make a generic request. Compose the scope dictionary and pass to the
        handler, return the result of the handler. Assume defaults for the
        query environment, which can be overridden using the arguments to the
        request.
        """
        scope = self._base_scope(**request)
        data = {}
        on_template_render = partial(store_rendered_templates, data)
        signal_uid = 'template-render-%s' % id(request)
        signals.template_rendered.connect(on_template_render, dispatch_uid=signal_uid)
        exception_uid = 'request-exception-%s' % id(request)
        got_request_exception.connect(self.store_exc_info, dispatch_uid=exception_uid)
        try:
            response = await self.handler(scope)
        finally:
            signals.template_rendered.disconnect(dispatch_uid=signal_uid)
            got_request_exception.disconnect(dispatch_uid=exception_uid)
        self.check_exception(response)
        response.client = self
        response.request = request
        response.templates = data.get('templates', [])
        response.context = data.get('context')
        response.json = partial(self._parse_json, response)
        urlconf = getattr(response.asgi_request, 'urlconf', None)
        response.resolver_match = SimpleLazyObject(lambda: resolve(request['path'], urlconf=urlconf))
        if response.context and len(response.context) == 1:
            response.context = response.context[0]
        if response.cookies:
            self.cookies.update(response.cookies)
        return response

    async def get(self, path, data=None, follow=False, secure=False, *, headers=None, **extra):
        """Request a response from the server using GET."""
        self.extra = extra
        self.headers = headers
        response = await super().get(path, data=data, secure=secure, headers=headers, **extra)
        if follow:
            response = await self._ahandle_redirects(response, data=data, headers=headers, **extra)
        return response

    async def post(self, path, data=None, content_type=MULTIPART_CONTENT, follow=False, secure=False, *, headers=None, **extra):
        """Request a response from the server using POST."""
        self.extra = extra
        self.headers = headers
        response = await super().post(path, data=data, content_type=content_type, secure=secure, headers=headers, **extra)
        if follow:
            response = await self._ahandle_redirects(response, data=data, content_type=content_type, headers=headers, **extra)
        return response

    async def head(self, path, data=None, follow=False, secure=False, *, headers=None, **extra):
        """Request a response from the server using HEAD."""
        self.extra = extra
        self.headers = headers
        response = await super().head(path, data=data, secure=secure, headers=headers, **extra)
        if follow:
            response = await self._ahandle_redirects(response, data=data, headers=headers, **extra)
        return response

    async def options(self, path, data='', content_type='application/octet-stream', follow=False, secure=False, *, headers=None, **extra):
        """Request a response from the server using OPTIONS."""
        self.extra = extra
        self.headers = headers
        response = await super().options(path, data=data, content_type=content_type, secure=secure, headers=headers, **extra)
        if follow:
            response = await self._ahandle_redirects(response, data=data, content_type=content_type, headers=headers, **extra)
        return response

    async def put(self, path, data='', content_type='application/octet-stream', follow=False, secure=False, *, headers=None, **extra):
        """Send a resource to the server using PUT."""
        self.extra = extra
        self.headers = headers
        response = await super().put(path, data=data, content_type=content_type, secure=secure, headers=headers, **extra)
        if follow:
            response = await self._ahandle_redirects(response, data=data, content_type=content_type, headers=headers, **extra)
        return response

    async def patch(self, path, data='', content_type='application/octet-stream', follow=False, secure=False, *, headers=None, **extra):
        """Send a resource to the server using PATCH."""
        self.extra = extra
        self.headers = headers
        response = await super().patch(path, data=data, content_type=content_type, secure=secure, headers=headers, **extra)
        if follow:
            response = await self._ahandle_redirects(response, data=data, content_type=content_type, headers=headers, **extra)
        return response

    async def delete(self, path, data='', content_type='application/octet-stream', follow=False, secure=False, *, headers=None, **extra):
        """Send a DELETE request to the server."""
        self.extra = extra
        self.headers = headers
        response = await super().delete(path, data=data, content_type=content_type, secure=secure, headers=headers, **extra)
        if follow:
            response = await self._ahandle_redirects(response, data=data, content_type=content_type, headers=headers, **extra)
        return response

    async def trace(self, path, data='', follow=False, secure=False, *, headers=None, **extra):
        """Send a TRACE request to the server."""
        self.extra = extra
        self.headers = headers
        response = await super().trace(path, data=data, secure=secure, headers=headers, **extra)
        if follow:
            response = await self._ahandle_redirects(response, data=data, headers=headers, **extra)
        return response

    async def _ahandle_redirects(self, response, data='', content_type='', headers=None, **extra):
        """
        Follow any redirects by requesting responses from the server using GET.
        """
        response.redirect_chain = []
        while response.status_code in REDIRECT_STATUS_CODES:
            redirect_chain = response.redirect_chain
            response = await self._follow_redirect(response, data=data, content_type=content_type, headers=headers, **extra)
            response.redirect_chain = redirect_chain
            self._ensure_redirects_not_cyclic(response)
        return response