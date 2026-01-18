from __future__ import annotations
from typing import Sequence
from . import Image
class QuadTransform(Transform):
    """
    Define a quad image transform.

    Maps a quadrilateral (a region defined by four corners) from the image to a
    rectangle of the given size.

    See :py:meth:`~PIL.Image.Image.transform`

    :param xy: An 8-tuple (x0, y0, x1, y1, x2, y2, x3, y3) which contain the
        upper left, lower left, lower right, and upper right corner of the
        source quadrilateral.
    """
    method = Image.Transform.QUAD