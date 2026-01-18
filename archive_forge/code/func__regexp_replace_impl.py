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
def _regexp_replace_impl(expr: ColumnElement[Any], op: OperatorType, pattern: Any, replacement: Any, flags: Optional[str], **kw: Any) -> ColumnElement[Any]:
    return BinaryExpression(expr, ExpressionClauseList._construct_for_list(operators.comma_op, type_api.NULLTYPE, coercions.expect(roles.BinaryElementRole, pattern, expr=expr, operator=operators.comma_op), coercions.expect(roles.BinaryElementRole, replacement, expr=expr, operator=operators.comma_op), group=False), op, modifiers={'flags': flags})