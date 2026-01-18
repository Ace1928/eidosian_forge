from __future__ import annotations
import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal
from
from
class _MSDate(sqltypes.Date):

    def bind_processor(self, dialect):

        def process(value):
            if type(value) == datetime.date:
                return datetime.datetime(value.year, value.month, value.day)
            else:
                return value
        return process
    _reg = re.compile('(\\d+)-(\\d+)-(\\d+)')

    def result_processor(self, dialect, coltype):

        def process(value):
            if isinstance(value, datetime.datetime):
                return value.date()
            elif isinstance(value, str):
                m = self._reg.match(value)
                if not m:
                    raise ValueError('could not parse %r as a date value' % (value,))
                return datetime.date(*[int(x or 0) for x in m.groups()])
            else:
                return value
        return process