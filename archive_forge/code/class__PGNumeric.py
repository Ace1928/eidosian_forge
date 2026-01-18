import decimal
import re
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .pg_catalog import _SpaceVector
from .pg_catalog import OIDVECTOR
from .types import CITEXT
from ... import exc
from ... import util
from ...engine import processors
from ...sql import sqltypes
from ...sql.elements import quoted_name
class _PGNumeric(sqltypes.Numeric):
    render_bind_cast = True

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