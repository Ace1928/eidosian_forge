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
class _OracleNumeric(sqltypes.Numeric):
    is_number = False

    def bind_processor(self, dialect):
        if self.scale == 0:
            return None
        elif self.asdecimal:
            processor = processors.to_decimal_processor_factory(decimal.Decimal, self._effective_decimal_return_scale)

            def process(value):
                if isinstance(value, (int, float)):
                    return processor(value)
                elif value is not None and value.is_infinite():
                    return float(value)
                else:
                    return value
            return process
        else:
            return processors.to_float

    def result_processor(self, dialect, coltype):
        return None

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