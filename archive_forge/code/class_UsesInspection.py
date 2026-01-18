from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class UsesInspection:
    __slots__ = ()
    _post_inspect: Literal[None] = None
    uses_inspection = True