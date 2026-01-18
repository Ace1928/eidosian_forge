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
class TransactionTestCase(SimpleTestCase):
    reset_sequences = False
    available_apps = None
    fixtures = None
    databases = {DEFAULT_DB_ALIAS}
    _disallowed_database_msg = 'Database %(operation)s to %(alias)r are not allowed in this test. Add %(alias)r to %(test)s.databases to ensure proper test isolation and silence this failure.'
    serialized_rollback = False

    def _pre_setup(self):
        """
        Perform pre-test setup:
        * If the class has an 'available_apps' attribute, restrict the app
          registry to these applications, then fire the post_migrate signal --
          it must run with the correct set of applications for the test case.
        * If the class has a 'fixtures' attribute, install those fixtures.
        """
        super()._pre_setup()
        if self.available_apps is not None:
            apps.set_available_apps(self.available_apps)
            setting_changed.send(sender=settings._wrapped.__class__, setting='INSTALLED_APPS', value=self.available_apps, enter=True)
            for db_name in self._databases_names(include_mirrors=False):
                emit_post_migrate_signal(verbosity=0, interactive=False, db=db_name)
        try:
            self._fixture_setup()
        except Exception:
            if self.available_apps is not None:
                apps.unset_available_apps()
                setting_changed.send(sender=settings._wrapped.__class__, setting='INSTALLED_APPS', value=settings.INSTALLED_APPS, enter=False)
            raise
        for db_name in self._databases_names(include_mirrors=False):
            connections[db_name].queries_log.clear()

    @classmethod
    def _databases_names(cls, include_mirrors=True):
        return [alias for alias in connections if alias in cls.databases and (include_mirrors or not connections[alias].settings_dict['TEST']['MIRROR'])]

    def _reset_sequences(self, db_name):
        conn = connections[db_name]
        if conn.features.supports_sequence_reset:
            sql_list = conn.ops.sequence_reset_by_name_sql(no_style(), conn.introspection.sequence_list())
            if sql_list:
                with transaction.atomic(using=db_name):
                    with conn.cursor() as cursor:
                        for sql in sql_list:
                            cursor.execute(sql)

    def _fixture_setup(self):
        for db_name in self._databases_names(include_mirrors=False):
            if self.reset_sequences:
                self._reset_sequences(db_name)
            if self.serialized_rollback and hasattr(connections[db_name], '_test_serialized_contents'):
                if self.available_apps is not None:
                    apps.unset_available_apps()
                connections[db_name].creation.deserialize_db_from_string(connections[db_name]._test_serialized_contents)
                if self.available_apps is not None:
                    apps.set_available_apps(self.available_apps)
            if self.fixtures:
                call_command('loaddata', *self.fixtures, verbosity=0, database=db_name)

    def _should_reload_connections(self):
        return True

    def _post_teardown(self):
        """
        Perform post-test things:
        * Flush the contents of the database to leave a clean slate. If the
          class has an 'available_apps' attribute, don't fire post_migrate.
        * Force-close the connection so the next test gets a clean cursor.
        """
        try:
            self._fixture_teardown()
            super()._post_teardown()
            if self._should_reload_connections():
                for conn in connections.all(initialized_only=True):
                    conn.close()
        finally:
            if self.available_apps is not None:
                apps.unset_available_apps()
                setting_changed.send(sender=settings._wrapped.__class__, setting='INSTALLED_APPS', value=settings.INSTALLED_APPS, enter=False)

    def _fixture_teardown(self):
        for db_name in self._databases_names(include_mirrors=False):
            inhibit_post_migrate = self.available_apps is not None or (self.serialized_rollback and hasattr(connections[db_name], '_test_serialized_contents'))
            call_command('flush', verbosity=0, interactive=False, database=db_name, reset_sequences=False, allow_cascade=self.available_apps is not None, inhibit_post_migrate=inhibit_post_migrate)

    def assertQuerysetEqual(self, *args, **kw):
        warnings.warn('assertQuerysetEqual() is deprecated in favor of assertQuerySetEqual().', category=RemovedInDjango51Warning, stacklevel=2)
        return self.assertQuerySetEqual(*args, **kw)

    def assertQuerySetEqual(self, qs, values, transform=None, ordered=True, msg=None):
        values = list(values)
        items = qs
        if transform is not None:
            items = map(transform, items)
        if not ordered:
            return self.assertDictEqual(Counter(items), Counter(values), msg=msg)
        if len(values) > 1 and hasattr(qs, 'ordered') and (not qs.ordered):
            raise ValueError('Trying to compare non-ordered queryset against more than one ordered value.')
        return self.assertEqual(list(items), values, msg=msg)

    def assertNumQueries(self, num, func=None, *args, using=DEFAULT_DB_ALIAS, **kwargs):
        conn = connections[using]
        context = _AssertNumQueriesContext(self, num, conn)
        if func is None:
            return context
        with context:
            func(*args, **kwargs)