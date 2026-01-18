from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class DMLSelectRole(SQLRole):
    """A SELECT statement embedded in DML, typically INSERT from SELECT"""
    __slots__ = ()
    _role_name = 'SELECT statement or equivalent textual object'