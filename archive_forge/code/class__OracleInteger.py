from __future__ import annotations
import decimal
import random
import re
from . import base as oracle
from .base import OracleCompiler
from .base import OracleDialect
from .base import OracleExecutionContext
from .types import _OracleDateLiteralRender
from ... import exc
from ... import util
from ...engine import cursor as _cursor
from ...engine import interfaces
from ...engine import processors
from ...sql import sqltypes
from ...sql._typing import is_sql_compiler
class _OracleInteger(sqltypes.Integer):

    def get_dbapi_type(self, dbapi):
        return int

    def _cx_oracle_var(self, dialect, cursor, arraysize=None):
        cx_Oracle = dialect.dbapi
        return cursor.var(cx_Oracle.STRING, 255, arraysize=arraysize if arraysize is not None else cursor.arraysize, outconverter=int)

    def _cx_oracle_outputtypehandler(self, dialect):

        def handler(cursor, name, default_type, size, precision, scale):
            return self._cx_oracle_var(dialect, cursor)
        return handler