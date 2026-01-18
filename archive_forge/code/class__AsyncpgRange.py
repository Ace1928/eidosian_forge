from __future__ import annotations
import collections
import decimal
import json as _py_json
import re
import time
from . import json
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import OID
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .base import REGCLASS
from .base import REGCONFIG
from .types import BIT
from .types import BYTEA
from .types import CITEXT
from ... import exc
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...engine import processors
from ...sql import sqltypes
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class _AsyncpgRange(ranges.AbstractSingleRangeImpl):

    def bind_processor(self, dialect):
        asyncpg_Range = dialect.dbapi.asyncpg.Range

        def to_range(value):
            if isinstance(value, ranges.Range):
                value = asyncpg_Range(value.lower, value.upper, lower_inc=value.bounds[0] == '[', upper_inc=value.bounds[1] == ']', empty=value.empty)
            return value
        return to_range

    def result_processor(self, dialect, coltype):

        def to_range(value):
            if value is not None:
                empty = value.isempty
                value = ranges.Range(value.lower, value.upper, bounds=f'{('[' if empty or value.lower_inc else '(')}{(']' if not empty and value.upper_inc else ')')}', empty=empty)
            return value
        return to_range