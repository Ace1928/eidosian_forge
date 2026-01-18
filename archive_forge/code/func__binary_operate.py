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
def _binary_operate(expr: ColumnElement[Any], op: OperatorType, obj: roles.BinaryElementRole[Any], *, reverse: bool=False, result_type: Optional[TypeEngine[_T]]=None, **kw: Any) -> OperatorExpression[_T]:
    coerced_obj = coercions.expect(roles.BinaryElementRole, obj, expr=expr, operator=op)
    if reverse:
        left, right = (coerced_obj, expr)
    else:
        left, right = (expr, coerced_obj)
    if result_type is None:
        op, result_type = left.comparator._adapt_expression(op, right.comparator)
    return OperatorExpression._construct_for_op(left, right, op, type_=result_type, modifiers=kw)