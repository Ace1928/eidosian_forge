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
def _deferredSkip(condition, reason, name):

    def decorator(test_func):
        nonlocal condition
        if not (isinstance(test_func, type) and issubclass(test_func, unittest.TestCase)):

            @wraps(test_func)
            def skip_wrapper(*args, **kwargs):
                if args and isinstance(args[0], unittest.TestCase) and (connection.alias not in getattr(args[0], 'databases', {})):
                    raise ValueError("%s cannot be used on %s as %s doesn't allow queries against the %r database." % (name, args[0], args[0].__class__.__qualname__, connection.alias))
                if condition():
                    raise unittest.SkipTest(reason)
                return test_func(*args, **kwargs)
            test_item = skip_wrapper
        else:
            test_item = test_func
            databases = getattr(test_item, 'databases', None)
            if not databases or connection.alias not in databases:

                def condition():
                    raise ValueError("%s cannot be used on %s as it doesn't allow queries against the '%s' database." % (name, test_item, connection.alias))
            skip = test_func.__dict__.get('__unittest_skip__')
            if isinstance(skip, CheckCondition):
                test_item.__unittest_skip__ = skip.add_condition(condition, reason)
            elif skip is not True:
                test_item.__unittest_skip__ = CheckCondition((condition, reason))
        return test_item
    return decorator