from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
class WrapperType(Protocol):

    def _new(self: WT, value: Any) -> WT:
        ...