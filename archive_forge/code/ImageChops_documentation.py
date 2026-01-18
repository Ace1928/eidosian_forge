from __future__ import annotations
from . import Image
Returns a copy of the image where data has been offset by the given
    distances. Data wraps around the edges. If ``yoffset`` is omitted, it
    is assumed to be equal to ``xoffset``.

    :param image: Input image.
    :param xoffset: The horizontal distance.
    :param yoffset: The vertical distance.  If omitted, both
        distances are set to the same value.
    :rtype: :py:class:`~PIL.Image.Image`
    