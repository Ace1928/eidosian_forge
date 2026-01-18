from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
class SQLORMOperations(SQLCoreOperations[_T_co], TypingOnly):
    __slots__ = ()
    if typing.TYPE_CHECKING:

        def of_type(self, class_: _EntityType[Any]) -> PropComparator[_T_co]:
            ...

        def and_(self, *criteria: _ColumnExpressionArgument[bool]) -> PropComparator[bool]:
            ...

        def any(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
            ...

        def has(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
            ...