from __future__ import annotations
from typing import Sequence
from . import Image
class PerspectiveTransform(Transform):
    """
    Define a perspective image transform.

    This function takes an 8-tuple (a, b, c, d, e, f, g, h). For each pixel
    (x, y) in the output image, the new value is taken from a position
    ((a x + b y + c) / (g x + h y + 1), (d x + e y + f) / (g x + h y + 1)) in
    the input image, rounded to nearest pixel.

    This function can be used to scale, translate, rotate, and shear the
    original image.

    See :py:meth:`.Image.transform`

    :param matrix: An 8-tuple (a, b, c, d, e, f, g, h).
    """
    method = Image.Transform.PERSPECTIVE