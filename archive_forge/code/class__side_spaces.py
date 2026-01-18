from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
@dataclass
class _side_spaces(ABC):
    """
    Base class to for spaces

    A *_space class should track the size taken up by all the objects that
    may fall on that side of the panel. The same name may appear in multiple
    side classes (e.g. legend).
    """
    pack: LayoutPack

    def __post_init__(self):
        self._calculate()

    def _calculate(self):
        """
        Calculate the space taken up by each artist
        """

    @property
    def total(self) -> float:
        """
        Total space
        """
        return sum((getattr(self, f.name) for f in fields(self)[1:]))

    def sum_upto(self, item: str) -> float:
        """
        Sum of space upto but not including item

        Sums starting at the edge of the figure i.e. the "plot_margin".
        """

        def _fields_upto(item: str) -> Generator[Field, None, None]:
            for f in fields(self)[1:]:
                if f.name == item:
                    break
                yield f
        return sum((getattr(self, f.name) for f in _fields_upto(item)))

    @cached_property
    def _legend_size(self) -> TupleFloat2:
        """
        Return size of legend in figure coordinates

        We need this to accurately justify the legend by proprotional
        values e.g. 0.2, instead of just left, right, top,  bottom &
        center.
        """
        return (0, 0)

    @cached_property
    def _legend_width(self) -> float:
        """
        Return width of legend in figure coordinates
        """
        return self._legend_size[0]

    @cached_property
    def _legend_height(self) -> float:
        """
        Return height of legend in figure coordinates
        """
        return self._legend_size[1]