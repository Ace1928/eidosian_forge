from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class CompoundElementRole(AllowsLambdaRole, SQLRole):
    """SELECT statements inside a CompoundSelect, e.g. UNION, EXTRACT, etc."""
    __slots__ = ()
    _role_name = 'SELECT construct for inclusion in a UNION or other set construct'