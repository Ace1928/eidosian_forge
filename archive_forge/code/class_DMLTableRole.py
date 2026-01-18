from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class DMLTableRole(FromClauseRole):
    __slots__ = ()
    _role_name = 'subject table for an INSERT, UPDATE or DELETE'