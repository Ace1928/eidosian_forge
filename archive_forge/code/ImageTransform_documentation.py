from __future__ import annotations
from typing import Sequence
from . import Image

    Define a mesh image transform.  A mesh transform consists of one or more
    individual quad transforms.

    See :py:meth:`~PIL.Image.Image.transform`

    :param data: A list of (bbox, quad) tuples.
    