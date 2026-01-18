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
class PGExecutionContext_pg8000(PGExecutionContext):

    def create_server_side_cursor(self):
        ident = 'c_%s_%s' % (hex(id(self))[2:], hex(_server_side_id())[2:])
        return ServerSideCursor(self._dbapi_connection.cursor(), ident)

    def pre_exec(self):
        if not self.compiled:
            return