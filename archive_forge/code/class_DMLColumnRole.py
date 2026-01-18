from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class DMLColumnRole(SQLRole):
    __slots__ = ()
    _role_name = 'SET/VALUES column expression or string key'