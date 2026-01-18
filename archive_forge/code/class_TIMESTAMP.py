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
class TIMESTAMP(sqltypes._Binary):
    """Implement the SQL Server TIMESTAMP type.

    Note this is **completely different** than the SQL Standard
    TIMESTAMP type, which is not supported by SQL Server.  It
    is a read-only datatype that does not support INSERT of values.

    .. versionadded:: 1.2

    .. seealso::

        :class:`_mssql.ROWVERSION`

    """
    __visit_name__ = 'TIMESTAMP'
    length = None

    def __init__(self, convert_int=False):
        """Construct a TIMESTAMP or ROWVERSION type.

        :param convert_int: if True, binary integer values will
         be converted to integers on read.

        .. versionadded:: 1.2

        """
        self.convert_int = convert_int

    def result_processor(self, dialect, coltype):
        super_ = super().result_processor(dialect, coltype)
        if self.convert_int:

            def process(value):
                if super_:
                    value = super_(value)
                if value is not None:
                    value = int(codecs.encode(value, 'hex'), 16)
                return value
            return process
        else:
            return super_