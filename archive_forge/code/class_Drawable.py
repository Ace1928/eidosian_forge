from __future__ import annotations
import os
import sys
from typing import TypeVar, Union
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import AbstractPointPen
class Drawable(Protocol):
    """Stand-in for an object that can draw itself with a given pen.

    See :mod:`fontTools.pens.basePen` for an introduction to pens.
    """

    def draw(self, pen: AbstractPen) -> None:
        ...