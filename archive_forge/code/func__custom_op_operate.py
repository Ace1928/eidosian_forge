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
def _custom_op_operate(expr: ColumnElement[Any], op: custom_op[Any], obj: Any, reverse: bool=False, result_type: Optional[TypeEngine[Any]]=None, **kw: Any) -> ColumnElement[Any]:
    if result_type is None:
        if op.return_type:
            result_type = op.return_type
        elif op.is_comparison:
            result_type = type_api.BOOLEANTYPE
    return _binary_operate(expr, op, obj, reverse=reverse, result_type=result_type, **kw)