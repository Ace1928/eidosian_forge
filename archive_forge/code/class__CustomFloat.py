from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
class _CustomFloat(Real, float):
    """Adds Real mixin while pretending to be a builtin float"""