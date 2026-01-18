from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class legend_artists:
    """
    Legend artists that are drawn on the figure
    """
    left: Optional[outside_legend] = None
    right: Optional[outside_legend] = None
    top: Optional[outside_legend] = None
    bottom: Optional[outside_legend] = None
    inside: list[inside_legend] = field(default_factory=list)

    @property
    def boxes(self) -> list[FlexibleAnchoredOffsetbox]:
        """
        Return list of all AnchoredOffsetboxes for the legends
        """
        lrtb = (l.box for l in (self.left, self.right, self.top, self.bottom) if l)
        inside = (l.box for l in self.inside)
        return list(itertools.chain([*lrtb, *inside]))