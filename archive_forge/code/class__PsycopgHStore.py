from __future__ import annotations
import decimal
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import PGDialect
from .base import PGExecutionContext
from .hstore import HSTORE
from .pg_catalog import _SpaceVector
from .pg_catalog import INT2VECTOR
from .pg_catalog import OIDVECTOR
from ... import exc
from ... import types as sqltypes
from ... import util
from ...engine import processors
class _PsycopgHStore(HSTORE):

    def bind_processor(self, dialect):
        if dialect._has_native_hstore:
            return None
        else:
            return super().bind_processor(dialect)

    def result_processor(self, dialect, coltype):
        if dialect._has_native_hstore:
            return None
        else:
            return super().result_processor(dialect, coltype)