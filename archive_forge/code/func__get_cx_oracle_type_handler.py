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
def _get_cx_oracle_type_handler(self, impl):
    if hasattr(impl, '_cx_oracle_outputtypehandler'):
        return impl._cx_oracle_outputtypehandler(self.dialect)
    else:
        return None