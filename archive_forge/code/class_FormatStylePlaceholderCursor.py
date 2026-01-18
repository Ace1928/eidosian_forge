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
class FormatStylePlaceholderCursor:
    """
    Django uses "format" (e.g. '%s') style placeholders, but Oracle uses ":var"
    style. This fixes it -- but note that if you want to use a literal "%s" in
    a query, you'll need to use "%%s".
    """
    charset = 'utf-8'

    def __init__(self, connection, database):
        self.cursor = connection.cursor()
        self.cursor.outputtypehandler = self._output_type_handler
        self.database = database

    @staticmethod
    def _output_number_converter(value):
        return decimal.Decimal(value) if '.' in value else int(value)

    @staticmethod
    def _get_decimal_converter(precision, scale):
        if scale == 0:
            return int
        context = decimal.Context(prec=precision)
        quantize_value = decimal.Decimal(1).scaleb(-scale)
        return lambda v: decimal.Decimal(v).quantize(quantize_value, context=context)

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

    def _format_params(self, params):
        try:
            return {k: OracleParam(v, self, True) for k, v in params.items()}
        except AttributeError:
            return tuple((OracleParam(p, self, True) for p in params))

    def _guess_input_sizes(self, params_list):
        if hasattr(params_list[0], 'keys'):
            sizes = {}
            for params in params_list:
                for k, value in params.items():
                    if value.input_size:
                        sizes[k] = value.input_size
            if sizes:
                self.setinputsizes(**sizes)
        else:
            sizes = [None] * len(params_list[0])
            for params in params_list:
                for i, value in enumerate(params):
                    if value.input_size:
                        sizes[i] = value.input_size
            if sizes:
                self.setinputsizes(*sizes)

    def _param_generator(self, params):
        if hasattr(params, 'items'):
            return {k: v.force_bytes for k, v in params.items()}
        else:
            return [p.force_bytes for p in params]

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

    def execute(self, query, params=None):
        query, params = self._fix_for_params(query, params, unify_by_values=True)
        self._guess_input_sizes([params])
        with wrap_oracle_errors():
            return self.cursor.execute(query, self._param_generator(params))

    def executemany(self, query, params=None):
        if not params:
            return None
        params_iter = iter(params)
        query, firstparams = self._fix_for_params(query, next(params_iter))
        formatted = [firstparams] + [self._format_params(p) for p in params_iter]
        self._guess_input_sizes(formatted)
        with wrap_oracle_errors():
            return self.cursor.executemany(query, [self._param_generator(p) for p in formatted])

    def close(self):
        try:
            self.cursor.close()
        except Database.InterfaceError:
            pass

    def var(self, *args):
        return VariableWrapper(self.cursor.var(*args))

    def arrayvar(self, *args):
        return VariableWrapper(self.cursor.arrayvar(*args))

    def __getattr__(self, attr):
        return getattr(self.cursor, attr)

    def __iter__(self):
        return iter(self.cursor)