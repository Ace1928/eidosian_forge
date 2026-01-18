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
def _get_decimal_converter(precision, scale):
    if scale == 0:
        return int
    context = decimal.Context(prec=precision)
    quantize_value = decimal.Decimal(1).scaleb(-scale)
    return lambda v: decimal.Decimal(v).quantize(quantize_value, context=context)