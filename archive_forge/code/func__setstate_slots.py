from __future__ import annotations
from typing import Any, Mapping
def _setstate_slots(self: Any, state: Any) -> None:
    for slot, value in state.items():
        setattr(self, slot, value)