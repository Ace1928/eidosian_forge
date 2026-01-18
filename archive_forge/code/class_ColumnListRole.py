from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class ColumnListRole(SQLRole):
    """Elements suitable for forming comma separated lists of expressions."""
    __slots__ = ()