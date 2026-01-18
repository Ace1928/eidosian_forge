from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class StatementRole(SQLRole):
    __slots__ = ()
    _role_name = 'Executable SQL or text() construct'
    if TYPE_CHECKING:

        @util.memoized_property
        def _propagate_attrs(self) -> _PropagateAttrsType:
            ...
    else:
        _propagate_attrs = util.EMPTY_DICT