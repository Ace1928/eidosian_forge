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
def _cx_oracle_outputtypehandler(self, dialect):
    cx_Oracle = dialect.dbapi

    def handler(cursor, name, default_type, size, precision, scale):
        outconverter = None
        if precision:
            if self.asdecimal:
                if default_type == cx_Oracle.NATIVE_FLOAT:
                    type_ = default_type
                    outconverter = decimal.Decimal
                else:
                    type_ = decimal.Decimal
            elif self.is_number and scale == 0:
                return None
            else:
                type_ = cx_Oracle.NATIVE_FLOAT
        elif self.asdecimal:
            if default_type == cx_Oracle.NATIVE_FLOAT:
                type_ = default_type
                outconverter = decimal.Decimal
            else:
                type_ = decimal.Decimal
        elif self.is_number and scale == 0:
            return None
        else:
            type_ = cx_Oracle.NATIVE_FLOAT
        return cursor.var(type_, 255, arraysize=cursor.arraysize, outconverter=outconverter)
    return handler