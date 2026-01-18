from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import type_api
from .elements import and_
from .elements import BinaryExpression
from .elements import ClauseElement
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ExpressionClauseList
from .elements import False_
from .elements import Null
from .elements import OperatorExpression
from .elements import or_
from .elements import True_
from .elements import UnaryExpression
from .operators import OperatorType
from .. import exc
from .. import util
def _in_impl(expr: ColumnElement[Any], op: OperatorType, seq_or_selectable: ClauseElement, negate_op: OperatorType, **kw: Any) -> ColumnElement[Any]:
    seq_or_selectable = coercions.expect(roles.InElementRole, seq_or_selectable, expr=expr, operator=op)
    if 'in_ops' in seq_or_selectable._annotations:
        op, negate_op = seq_or_selectable._annotations['in_ops']
    return _boolean_compare(expr, op, seq_or_selectable, negate_op=negate_op, **kw)