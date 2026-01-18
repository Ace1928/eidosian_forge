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
class _PGExecutionContext_common_psycopg(PGExecutionContext):

    def create_server_side_cursor(self):
        ident = 'c_%s_%s' % (hex(id(self))[2:], hex(_server_side_id())[2:])
        return self._dbapi_connection.cursor(ident)