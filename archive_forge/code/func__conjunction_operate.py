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
def _conjunction_operate(expr: ColumnElement[Any], op: OperatorType, other: Any, **kw: Any) -> ColumnElement[Any]:
    if op is operators.and_:
        return and_(expr, other)
    elif op is operators.or_:
        return or_(expr, other)
    else:
        raise NotImplementedError()