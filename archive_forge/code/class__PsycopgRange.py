from __future__ import annotations
import logging
import re
from typing import cast
from typing import TYPE_CHECKING
from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import INTERVAL
from .base import PGCompiler
from .base import PGIdentifierPreparer
from .base import REGCONFIG
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .types import CITEXT
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...sql import sqltypes
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class _PsycopgRange(ranges.AbstractSingleRangeImpl):

    def bind_processor(self, dialect):
        psycopg_Range = cast(PGDialect_psycopg, dialect)._psycopg_Range

        def to_range(value):
            if isinstance(value, ranges.Range):
                value = psycopg_Range(value.lower, value.upper, value.bounds, value.empty)
            return value
        return to_range

    def result_processor(self, dialect, coltype):

        def to_range(value):
            if value is not None:
                value = ranges.Range(value._lower, value._upper, bounds=value._bounds if value._bounds else '[)', empty=not value._bounds)
            return value
        return to_range