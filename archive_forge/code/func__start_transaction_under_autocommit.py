import datetime
import decimal
import warnings
from collections.abc import Mapping
from itertools import chain, tee
from sqlite3 import dbapi2 as Database
from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.db.backends.base.base import BaseDatabaseWrapper
from django.utils.asyncio import async_unsafe
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.regex_helper import _lazy_re_compile
from ._functions import register as register_functions
from .client import DatabaseClient
from .creation import DatabaseCreation
from .features import DatabaseFeatures
from .introspection import DatabaseIntrospection
from .operations import DatabaseOperations
from .schema import DatabaseSchemaEditor
def _start_transaction_under_autocommit(self):
    """
        Start a transaction explicitly in autocommit mode.

        Staying in autocommit mode works around a bug of sqlite3 that breaks
        savepoints when autocommit is disabled.
        """
    self.cursor().execute('BEGIN')