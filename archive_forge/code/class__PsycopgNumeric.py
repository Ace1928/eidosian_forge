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
class _PsycopgNumeric(sqltypes.Numeric):

    def bind_processor(self, dialect):
        return None

    def result_processor(self, dialect, coltype):
        if self.asdecimal:
            if coltype in _FLOAT_TYPES:
                return processors.to_decimal_processor_factory(decimal.Decimal, self._effective_decimal_return_scale)
            elif coltype in _DECIMAL_TYPES or coltype in _INT_TYPES:
                return None
            else:
                raise exc.InvalidRequestError('Unknown PG numeric type: %d' % coltype)
        elif coltype in _FLOAT_TYPES:
            return None
        elif coltype in _DECIMAL_TYPES or coltype in _INT_TYPES:
            return processors.to_float
        else:
            raise exc.InvalidRequestError('Unknown PG numeric type: %d' % coltype)