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
class _OracleBinary(_LOBDataType, sqltypes.LargeBinary):

    def get_dbapi_type(self, dbapi):
        return dbapi.DB_TYPE_RAW

    def bind_processor(self, dialect):
        return None

    def result_processor(self, dialect, coltype):
        if not dialect.auto_convert_lobs:
            return None
        else:
            return super().result_processor(dialect, coltype)