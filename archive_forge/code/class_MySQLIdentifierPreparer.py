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
class MySQLIdentifierPreparer(compiler.IdentifierPreparer):
    reserved_words = RESERVED_WORDS_MYSQL

    def __init__(self, dialect, server_ansiquotes=False, **kw):
        if not server_ansiquotes:
            quote = '`'
        else:
            quote = '"'
        super().__init__(dialect, initial_quote=quote, escape_quote=quote)

    def _quote_free_identifiers(self, *ids):
        """Unilaterally identifier-quote any number of strings."""
        return tuple([self.quote_identifier(i) for i in ids if i is not None])