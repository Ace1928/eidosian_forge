from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING
@dataclasses.dataclass(order=True, frozen=True)
class OutputKey:
    label: Hashable
    position: int