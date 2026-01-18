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
class _CXOracleDate(oracle._OracleDate):

    def bind_processor(self, dialect):
        return None

    def result_processor(self, dialect, coltype):

        def process(value):
            if value is not None:
                return value.date()
            else:
                return value
        return process