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
def _setup_and_call(self, result, debug=False):
    """
        Perform the following in order: pre-setup, run test, post-teardown,
        skipping pre/post hooks if test is set to be skipped.

        If debug=True, reraise any errors in setup and use super().debug()
        instead of __call__() to run the test.
        """
    testMethod = getattr(self, self._testMethodName)
    skipped = getattr(self.__class__, '__unittest_skip__', False) or getattr(testMethod, '__unittest_skip__', False)
    if iscoroutinefunction(testMethod):
        setattr(self, self._testMethodName, async_to_sync(testMethod))
    if not skipped:
        try:
            self._pre_setup()
        except Exception:
            if debug:
                raise
            result.addError(self, sys.exc_info())
            return
    if debug:
        super().debug()
    else:
        super().__call__(result)
    if not skipped:
        try:
            self._post_teardown()
        except Exception:
            if debug:
                raise
            result.addError(self, sys.exc_info())
            return