from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import attributes
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.attributes import QueryableAttribute
from ..sql import roles
from ..sql._typing import is_has_clause_element
from ..sql.elements import ColumnElement
from ..sql.elements import SQLCoreOperations
from ..util.typing import Concatenate
from ..util.typing import Literal
from ..util.typing import ParamSpec
from ..util.typing import Protocol
from ..util.typing import Self
class ExprComparator(Comparator[_T]):

    def __init__(self, cls: Type[Any], expression: Union[_HasClauseElement[_T], SQLColumnExpression[_T]], hybrid: hybrid_property[_T]):
        self.cls = cls
        self.expression = expression
        self.hybrid = hybrid

    def __getattr__(self, key: str) -> Any:
        return getattr(self.expression, key)

    @util.ro_non_memoized_property
    def info(self) -> _InfoType:
        return self.hybrid.info

    def _bulk_update_tuples(self, value: Any) -> Sequence[Tuple[_DMLColumnArgument, Any]]:
        if isinstance(self.expression, attributes.QueryableAttribute):
            return self.expression._bulk_update_tuples(value)
        elif self.hybrid.update_expr is not None:
            return self.hybrid.update_expr(self.cls, value)
        else:
            return [(self.expression, value)]

    @util.non_memoized_property
    def property(self) -> MapperProperty[_T]:
        return self.expression.property

    def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[Any]:
        return op(self.expression, *other, **kwargs)

    def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[Any]:
        return op(other, self.expression, **kwargs)