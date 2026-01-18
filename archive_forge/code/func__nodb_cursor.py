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
@contextmanager
def _nodb_cursor(self):
    cursor = None
    try:
        with super()._nodb_cursor() as cursor:
            yield cursor
    except (Database.DatabaseError, WrappedDatabaseError):
        if cursor is not None:
            raise
        warnings.warn("Normally Django will use a connection to the 'postgres' database to avoid running initialization queries against the production database when it's not needed (for example, when running tests). Django was unable to create a connection to the 'postgres' database and will use the first PostgreSQL database instead.", RuntimeWarning)
        for connection in connections.all():
            if connection.vendor == 'postgresql' and connection.settings_dict['NAME'] != 'postgres':
                conn = self.__class__({**self.settings_dict, 'NAME': connection.settings_dict['NAME']}, alias=self.alias)
                try:
                    with conn.cursor() as cursor:
                        yield cursor
                finally:
                    conn.close()
                break
        else:
            raise