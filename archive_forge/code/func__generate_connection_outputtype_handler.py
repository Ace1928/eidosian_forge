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
def _generate_connection_outputtype_handler(self):
    """establish the default outputtypehandler established at the
        connection level.

        """
    dialect = self
    cx_Oracle = dialect.dbapi
    number_handler = _OracleNUMBER(asdecimal=True)._cx_oracle_outputtypehandler(dialect)
    float_handler = _OracleNUMBER(asdecimal=False)._cx_oracle_outputtypehandler(dialect)

    def output_type_handler(cursor, name, default_type, size, precision, scale):
        if default_type == cx_Oracle.NUMBER and default_type is not cx_Oracle.NATIVE_FLOAT:
            if not dialect.coerce_to_decimal:
                return None
            elif precision == 0 and scale in (0, -127):
                return cursor.var(cx_Oracle.STRING, 255, outconverter=dialect._detect_decimal, arraysize=cursor.arraysize)
            elif precision and scale > 0:
                return number_handler(cursor, name, default_type, size, precision, scale)
            else:
                return float_handler(cursor, name, default_type, size, precision, scale)
        elif dialect._cursor_var_unicode_kwargs and default_type in (cx_Oracle.STRING, cx_Oracle.FIXED_CHAR) and (default_type is not cx_Oracle.CLOB) and (default_type is not cx_Oracle.NCLOB):
            return cursor.var(str, size, cursor.arraysize, **dialect._cursor_var_unicode_kwargs)
        elif dialect.auto_convert_lobs and default_type in (cx_Oracle.CLOB, cx_Oracle.NCLOB):
            return cursor.var(cx_Oracle.DB_TYPE_NVARCHAR, _CX_ORACLE_MAGIC_LOB_SIZE, cursor.arraysize, **dialect._cursor_var_unicode_kwargs)
        elif dialect.auto_convert_lobs and default_type in (cx_Oracle.BLOB,):
            return cursor.var(cx_Oracle.DB_TYPE_RAW, _CX_ORACLE_MAGIC_LOB_SIZE, cursor.arraysize)
    return output_type_handler