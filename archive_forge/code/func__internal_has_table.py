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
@reflection.cache
def _internal_has_table(self, connection, tablename, owner, **kw):
    if tablename.startswith('#'):
        return bool(connection.scalar(text("SELECT object_id(:table_name, 'U')"), {'table_name': f'tempdb.dbo.[{tablename}]'}))
    else:
        tables = ischema.tables
        s = sql.select(tables.c.table_name).where(sql.and_(sql.or_(tables.c.table_type == 'BASE TABLE', tables.c.table_type == 'VIEW'), tables.c.table_name == tablename))
        if owner:
            s = s.where(tables.c.table_schema == owner)
        c = connection.execute(s)
        return c.first() is not None