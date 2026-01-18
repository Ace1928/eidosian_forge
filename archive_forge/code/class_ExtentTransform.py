from __future__ import annotations
from typing import Sequence
from . import Image
class ExtentTransform(Transform):
    """
    Define a transform to extract a subregion from an image.

    Maps a rectangle (defined by two corners) from the image to a rectangle of
    the given size. The resulting image will contain data sampled from between
    the corners, such that (x0, y0) in the input image will end up at (0,0) in
    the output image, and (x1, y1) at size.

    This method can be used to crop, stretch, shrink, or mirror an arbitrary
    rectangle in the current image. It is slightly slower than crop, but about
    as fast as a corresponding resize operation.

    See :py:meth:`~PIL.Image.Image.transform`

    :param bbox: A 4-tuple (x0, y0, x1, y1) which specifies two points in the
        input image's coordinate system. See :ref:`coordinate-system`.
    """
    method = Image.Transform.EXTENT