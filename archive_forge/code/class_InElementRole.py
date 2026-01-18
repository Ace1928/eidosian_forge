from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class InElementRole(SQLRole):
    __slots__ = ()
    _role_name = 'IN expression list, SELECT construct, or bound parameter object'