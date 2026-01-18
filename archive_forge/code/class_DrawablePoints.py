from __future__ import annotations
import os
import sys
from typing import TypeVar, Union
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import AbstractPointPen
class DrawablePoints(Protocol):
    """Stand-in for an object that can draw its points with a given pen.

    See :mod:`fontTools.pens.pointPen` for an introduction to point pens.
    """

    def drawPoints(self, pen: AbstractPointPen) -> None:
        ...