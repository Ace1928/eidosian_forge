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
def _generate_cursor_outputtype_handler(self):
    output_handlers = {}
    for keyname, name, objects, type_ in self.compiled._result_columns:
        handler = type_._cached_custom_processor(self.dialect, 'cx_oracle_outputtypehandler', self._get_cx_oracle_type_handler)
        if handler:
            denormalized_name = self.dialect.denormalize_name(keyname)
            output_handlers[denormalized_name] = handler
    if output_handlers:
        default_handler = self._dbapi_connection.outputtypehandler

        def output_type_handler(cursor, name, default_type, size, precision, scale):
            if name in output_handlers:
                return output_handlers[name](cursor, name, default_type, size, precision, scale)
            else:
                return default_handler(cursor, name, default_type, size, precision, scale)
        self.cursor.outputtypehandler = output_type_handler