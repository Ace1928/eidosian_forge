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
class PGIdentifierPreparer_pg8000(PGIdentifierPreparer):

    def __init__(self, *args, **kwargs):
        PGIdentifierPreparer.__init__(self, *args, **kwargs)
        self._double_percents = False