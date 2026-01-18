from __future__ import annotations
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import Numeric
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import BIGINT
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import NULLTYPE
from .base import alter_column
from .base import alter_table
from .base import AlterColumn
from .base import ColumnComment
from .base import format_column_name
from .base import format_table_name
from .base import format_type
from .base import IdentityColumnDefault
from .base import RenameTable
from .impl import ComparisonResult
from .impl import DefaultImpl
from .. import util
from ..autogenerate import render
from ..operations import ops
from ..operations import schemaobj
from ..operations.base import BatchOperations
from ..operations.base import Operations
from ..util import sqla_compat
from ..util.sqla_compat import compiles
def _cleanup_index_expr(self, index: Index, expr: str) -> str:
    expr = expr.lower().replace('"', '').replace("'", '')
    if index.table is not None:
        expr = expr.replace(f'{index.table.name.lower()}.', '')
    if '::' in expr:
        expr = re.sub('(::[\\w ]+\\w)', '', expr)
    while expr and expr[0] == '(' and (expr[-1] == ')'):
        expr = expr[1:-1]
    for rs in self._default_modifiers_re:
        if (match := rs.search(expr)):
            start, end = match.span(1)
            expr = expr[:start] + expr[end:]
            break
    while expr and expr[0] == '(' and (expr[-1] == ')'):
        expr = expr[1:-1]
    cast_re = re.compile('cast\\s*\\(')
    if cast_re.match(expr):
        expr = cast_re.sub('', expr)
        expr = re.sub('as\\s+[^)]+\\)', '', expr)
    expr = expr.replace(' ', '')
    return expr