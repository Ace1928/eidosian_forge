from __future__ import annotations
import logging # isort:skip
from typing import Any, ClassVar, Literal
from ..core.properties import (
from ..core.property.aliases import CoordinateLike
from ..model import Model
class BoxNodes:
    """ Provider of box nodes for box-like models. """

    def __init__(self, target: Model | ImplicitTarget) -> None:
        self.target = target

    def _node(self, symbol: str) -> Node:
        return Node(target=self.target, symbol=symbol)

    @property
    def left(self) -> Node:
        return self._node('left')

    @property
    def right(self) -> Node:
        return self._node('right')

    @property
    def top(self) -> Node:
        return self._node('top')

    @property
    def bottom(self) -> Node:
        return self._node('bottom')

    @property
    def top_left(self) -> Node:
        return self._node('top_left')

    @property
    def top_center(self) -> Node:
        return self._node('top_center')

    @property
    def top_right(self) -> Node:
        return self._node('top_right')

    @property
    def center_left(self) -> Node:
        return self._node('center_left')

    @property
    def center(self) -> Node:
        return self._node('center')

    @property
    def center_right(self) -> Node:
        return self._node('center_right')

    @property
    def bottom_left(self) -> Node:
        return self._node('bottom_left')

    @property
    def bottom_center(self) -> Node:
        return self._node('bottom_center')

    @property
    def bottom_right(self) -> Node:
        return self._node('bottom_right')

    @property
    def width(self) -> Node:
        return self._node('width')

    @property
    def height(self) -> Node:
        return self._node('height')