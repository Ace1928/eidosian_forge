from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
class _CustomList(MutableSequence, list):
    """Adds MutableSequence mixin while pretending to be a builtin list"""