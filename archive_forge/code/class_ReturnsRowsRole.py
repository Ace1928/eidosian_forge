from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class ReturnsRowsRole(SQLRole):
    __slots__ = ()
    _role_name = 'Row returning expression such as a SELECT, a FROM clause, or an INSERT/UPDATE/DELETE with RETURNING'