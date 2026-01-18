import datetime
import decimal
import os
import platform
from contextlib import contextmanager
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.backends.oracle.oracledb_any import oracledb as Database
from django.db.backends.utils import debug_transaction
from django.utils.asyncio import async_unsafe
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property
from .client import DatabaseClient  # NOQA
from .creation import DatabaseCreation  # NOQA
from .features import DatabaseFeatures  # NOQA
from .introspection import DatabaseIntrospection  # NOQA
from .operations import DatabaseOperations  # NOQA
from .schema import DatabaseSchemaEditor  # NOQA
from .utils import Oracle_datetime, dsn  # NOQA
from .validation import DatabaseValidation  # NOQA
@staticmethod
def _output_type_handler(cursor, name, defaultType, length, precision, scale):
    """
        Called for each db column fetched from cursors. Return numbers as the
        appropriate Python type, and NCLOB with JSON as strings.
        """
    if defaultType == Database.NUMBER:
        if scale == -127:
            if precision == 0:
                outconverter = FormatStylePlaceholderCursor._output_number_converter
            else:
                outconverter = float
        elif precision > 0:
            outconverter = FormatStylePlaceholderCursor._get_decimal_converter(precision, scale)
        else:
            outconverter = FormatStylePlaceholderCursor._output_number_converter
        return cursor.var(Database.STRING, size=255, arraysize=cursor.arraysize, outconverter=outconverter)
    elif defaultType == Database.DB_TYPE_NCLOB:
        return cursor.var(Database.DB_TYPE_NCLOB, arraysize=cursor.arraysize)