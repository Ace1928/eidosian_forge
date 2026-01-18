import asyncio
import threading
import warnings
from contextlib import contextmanager
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import DatabaseError as WrappedDatabaseError
from django.db import connections
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.backends.utils import CursorDebugWrapper as BaseCursorDebugWrapper
from django.utils.asyncio import async_unsafe
from django.utils.functional import cached_property
from django.utils.safestring import SafeString
from django.utils.version import get_version_tuple
from .psycopg_any import IsolationLevel, is_psycopg3  # NOQA isort:skip
from .client import DatabaseClient  # NOQA isort:skip
from .creation import DatabaseCreation  # NOQA isort:skip
from .features import DatabaseFeatures  # NOQA isort:skip
from .introspection import DatabaseIntrospection  # NOQA isort:skip
from .operations import DatabaseOperations  # NOQA isort:skip
from .schema import DatabaseSchemaEditor  # NOQA isort:skip
class CursorMixin:
    """
        A subclass of psycopg cursor implementing callproc.
        """

    def callproc(self, name, args=None):
        if not isinstance(name, sql.Identifier):
            name = sql.Identifier(name)
        qparts = [sql.SQL('SELECT * FROM '), name, sql.SQL('(')]
        if args:
            for item in args:
                qparts.append(sql.Literal(item))
                qparts.append(sql.SQL(','))
            del qparts[-1]
        qparts.append(sql.SQL(')'))
        stmt = sql.Composed(qparts)
        self.execute(stmt)
        return args