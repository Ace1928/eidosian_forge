import difflib
import json
import logging
import posixpath
import sys
import threading
import unittest
import warnings
from collections import Counter
from contextlib import contextmanager
from copy import copy, deepcopy
from difflib import get_close_matches
from functools import wraps
from unittest.suite import _DebugResult
from unittest.util import safe_repr
from urllib.parse import (
from urllib.request import url2pathname
from asgiref.sync import async_to_sync, iscoroutinefunction
from django.apps import apps
from django.conf import settings
from django.core import mail
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.core.files import locks
from django.core.handlers.wsgi import WSGIHandler, get_path_info
from django.core.management import call_command
from django.core.management.color import no_style
from django.core.management.sql import emit_post_migrate_signal
from django.core.servers.basehttp import ThreadedWSGIServer, WSGIRequestHandler
from django.core.signals import setting_changed
from django.db import DEFAULT_DB_ALIAS, connection, connections, transaction
from django.forms.fields import CharField
from django.http import QueryDict
from django.http.request import split_domain_port, validate_host
from django.test.client import AsyncClient, Client
from django.test.html import HTMLParseError, parse_html
from django.test.signals import template_rendered
from django.test.utils import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import classproperty
from django.views.static import serve
class FSFilesHandler(WSGIHandler):
    """
    WSGI middleware that intercepts calls to a directory, as defined by one of
    the *_ROOT settings, and serves those files, publishing them under *_URL.
    """

    def __init__(self, application):
        self.application = application
        self.base_url = urlparse(self.get_base_url())
        super().__init__()

    def _should_handle(self, path):
        """
        Check if the path should be handled. Ignore the path if:
        * the host is provided as part of the base_url
        * the request's path isn't under the media path (or equal)
        """
        return path.startswith(self.base_url[2]) and (not self.base_url[1])

    def file_path(self, url):
        """Return the relative path to the file on disk for the given URL."""
        relative_url = url.removeprefix(self.base_url[2])
        return url2pathname(relative_url)

    def get_response(self, request):
        from django.http import Http404
        if self._should_handle(request.path):
            try:
                return self.serve(request)
            except Http404:
                pass
        return super().get_response(request)

    def serve(self, request):
        os_rel_path = self.file_path(request.path)
        os_rel_path = posixpath.normpath(unquote(os_rel_path))
        final_rel_path = os_rel_path.replace('\\', '/').lstrip('/')
        return serve(request, final_rel_path, document_root=self.get_base_dir())

    def __call__(self, environ, start_response):
        if not self._should_handle(get_path_info(environ)):
            return self.application(environ, start_response)
        return super().__call__(environ, start_response)