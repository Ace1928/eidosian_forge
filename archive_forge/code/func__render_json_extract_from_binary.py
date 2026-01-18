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
def _render_json_extract_from_binary(self, binary, operator, **kw):
    if binary.type._type_affinity is sqltypes.JSON:
        return 'JSON_QUERY(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
    case_expression = 'CASE JSON_VALUE(%s, %s) WHEN NULL THEN NULL' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
    if binary.type._type_affinity is sqltypes.Integer:
        type_expression = 'ELSE CAST(JSON_VALUE(%s, %s) AS INTEGER)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
    elif binary.type._type_affinity is sqltypes.Numeric:
        type_expression = 'ELSE CAST(JSON_VALUE(%s, %s) AS %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw), 'FLOAT' if isinstance(binary.type, sqltypes.Float) else 'NUMERIC(%s, %s)' % (binary.type.precision, binary.type.scale))
    elif binary.type._type_affinity is sqltypes.Boolean:
        type_expression = "WHEN 'true' THEN 1 WHEN 'false' THEN 0 ELSE NULL"
    elif binary.type._type_affinity is sqltypes.String:
        type_expression = 'ELSE JSON_VALUE(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
    else:
        type_expression = 'ELSE JSON_QUERY(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
    return case_expression + ' ' + type_expression + ' END'