from __future__ import annotations
import asyncio
import base64
import struct
from io import BytesIO
from pathlib import PurePath
from typing import (
import param
from ..models import PDF as _BkPDF
from ..util import isfile, isurl
from .markup import HTMLBasePane, escape
class JPG(ImageBase):
    """
    The `JPG` pane embeds a .jpg or .jpeg image file in a panel if
    provided a local path, or will link to a remote image if provided
    a URL.

    Reference: https://panel.holoviz.org/reference/panes/JPG.html

    :Example:

    >>> JPG(
    ...     'https://www.gstatic.com/webp/gallery/4.sm.jpg',
    ...     alt_text='A nice tree',
    ...     link_url='https://en.wikipedia.org/wiki/JPEG',
    ...     width=500
    ... )
    """
    filetype: ClassVar[str] = 'jpeg'
    _extensions: ClassVar[Tuple[str, ...]] = ('jpeg', 'jpg')

    @classmethod
    def _imgshape(cls, data):
        import struct
        b = BytesIO(data)
        b.read(2)
        c = b.read(1)
        while c and ord(c) != 218:
            while ord(c) != 255:
                c = b.read(1)
            while ord(c) == 255:
                c = b.read(1)
            if ord(c) >= 192 and ord(c) <= 195:
                b.read(3)
                h, w = struct.unpack('>HH', b.read(4))
                break
            else:
                b.read(int(struct.unpack('>H', b.read(2))[0]) - 2)
            c = b.read(1)
        return (int(w), int(h))