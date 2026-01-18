from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class NoReferencedTableError(NoReferenceError):
    """Raised by ``ForeignKey`` when the referred ``Table`` cannot be
    located.

    """

    def __init__(self, message: str, tname: str):
        NoReferenceError.__init__(self, message)
        self.table_name = tname

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (self.__class__, (self.args[0], self.table_name))