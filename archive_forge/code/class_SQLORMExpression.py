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
class SQLORMExpression(SQLORMOperations[_T_co], SQLColumnExpression[_T_co], TypingOnly):
    """A type that may be used to indicate any ORM-level attribute or
    object that acts in place of one, in the context of SQL expression
    construction.

    :class:`.SQLORMExpression` extends from the Core
    :class:`.SQLColumnExpression` to add additional SQL methods that are ORM
    specific, such as :meth:`.PropComparator.of_type`, and is part of the bases
    for :class:`.InstrumentedAttribute`. It may be used in :pep:`484` typing to
    indicate arguments or return values that should behave as ORM-level
    attribute expressions.

    .. versionadded:: 2.0.0b4


    """
    __slots__ = ()