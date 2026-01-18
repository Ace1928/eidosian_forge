from __future__ import annotations
from abc import ABC
from dataclasses import asdict, dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from .._utils import ensure_xy_location, get_opposite_side
from .._utils.registry import Register
from ..themes.theme import theme as Theme
@cached_property
def _position_inside(self) -> SidePosition | TupleFloat2:
    pos = self.theme.getp('legend_position_inside')
    if isinstance(pos, tuple):
        return pos
    just = self.theme.getp('legend_justification_inside', (0.5, 0.5))
    return ensure_xy_location(just)