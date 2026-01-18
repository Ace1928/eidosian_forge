from the proposed insertion.   These values are normally specified using
from __future__ import annotations
from array import array as _array
from collections import defaultdict
from itertools import compress
import re
from typing import cast
from . import reflection as _reflection
from .enumerated import ENUM
from .enumerated import SET
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from .reserved_words import RESERVED_WORDS_MARIADB
from .reserved_words import RESERVED_WORDS_MYSQL
from .types import _FloatType
from .types import _IntegerType
from .types import _MatchType
from .types import _NumericType
from .types import _StringType
from .types import BIGINT
from .types import BIT
from .types import CHAR
from .types import DATETIME
from .types import DECIMAL
from .types import DOUBLE
from .types import FLOAT
from .types import INTEGER
from .types import LONGBLOB
from .types import LONGTEXT
from .types import MEDIUMBLOB
from .types import MEDIUMINT
from .types import MEDIUMTEXT
from .types import NCHAR
from .types import NUMERIC
from .types import NVARCHAR
from .types import REAL
from .types import SMALLINT
from .types import TEXT
from .types import TIME
from .types import TIMESTAMP
from .types import TINYBLOB
from .types import TINYINT
from .types import TINYTEXT
from .types import VARCHAR
from .types import YEAR
from ... import exc
from ... import literal_column
from ... import log
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import functions
from ...sql import operators
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.compiler import SQLCompiler
from ...sql.schema import SchemaConst
from ...types import BINARY
from ...types import BLOB
from ...types import BOOLEAN
from ...types import DATE
from ...types import UUID
from ...types import VARBINARY
from ...util import topological
class _DecodingRow:
    """Return unicode-decoded values based on type inspection.

    Smooth over data type issues (esp. with alpha driver versions) and
    normalize strings as Unicode regardless of user-configured driver
    encoding settings.

    """
    _encoding_compat = {'koi8r': 'koi8_r', 'koi8u': 'koi8_u', 'utf16': 'utf-16-be', 'utf8mb4': 'utf8', 'utf8mb3': 'utf8', 'eucjpms': 'ujis'}

    def __init__(self, rowproxy, charset):
        self.rowproxy = rowproxy
        self.charset = self._encoding_compat.get(charset, charset)

    def __getitem__(self, index):
        item = self.rowproxy[index]
        if isinstance(item, _array):
            item = item.tostring()
        if self.charset and isinstance(item, bytes):
            return item.decode(self.charset)
        else:
            return item

    def __getattr__(self, attr):
        item = getattr(self.rowproxy, attr)
        if isinstance(item, _array):
            item = item.tostring()
        if self.charset and isinstance(item, bytes):
            return item.decode(self.charset)
        else:
            return item