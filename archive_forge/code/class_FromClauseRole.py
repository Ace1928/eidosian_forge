from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class FromClauseRole(ColumnsClauseRole, JoinTargetRole):
    __slots__ = ()
    _role_name = 'FROM expression, such as a Table or alias() object'
    _is_subquery = False
    named_with_column: bool