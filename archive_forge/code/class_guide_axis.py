from __future__ import annotations
from dataclasses import dataclass, field
from .guide import guide
@dataclass
class guide_axis(guide):
    """
    Axis
    """
    available_aes: set[str] = field(init=False, default_factory=lambda: {'x', 'y'})