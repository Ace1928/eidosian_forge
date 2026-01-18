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
def _fix_for_params(self, query, params, unify_by_values=False):
    if query.endswith(';') or query.endswith('/'):
        query = query[:-1]
    if params is None:
        params = []
    elif hasattr(params, 'keys'):
        args = {k: ':%s' % k for k in params}
        query %= args
    elif unify_by_values and params:
        param_types = [(type(param), param) for param in params]
        params_dict = {param_type: ':arg%d' % i for i, param_type in enumerate(dict.fromkeys(param_types))}
        args = [params_dict[param_type] for param_type in param_types]
        params = {placeholder: param for (_, param), placeholder in params_dict.items()}
        query %= tuple(args)
    else:
        args = [':arg%d' % i for i in range(len(params))]
        query %= tuple(args)
    return (query, self._format_params(params))