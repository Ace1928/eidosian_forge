from __future__ import annotations
import warnings
from typing import Optional
from attrs import define, field
from fontTools.misc.transform import Identity, Transform
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from ufoLib2.objects.misc import BoundingBox
from ufoLib2.serde import serde
from ufoLib2.typing import GlyphSet
from .misc import _convert_transform, getBounds, getControlBounds
Draws points of component with given point pen.