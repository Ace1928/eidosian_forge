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
class WebP(ImageBase):
    """
    The `WebP` pane embeds a .webp image file in a panel if
    provided a local path, or will link to a remote image if provided
    a URL.

    Reference: https://panel.holoviz.org/reference/panes/WebP.html

    :Example:

    >>> WebP(
    ...     'https://assets.holoviz.org/panel/samples/webp_sample.webp',
    ...     alt_text='A nice tree',
    ...     link_url='https://en.wikipedia.org/wiki/WebP',
    ...     width=500,
    ...     caption='A nice tree'
    ... )
    """
    filetype: ClassVar[str] = 'webp'
    _extensions: ClassVar[Tuple[str, ...]] = ('webp',)

    @classmethod
    def _imgshape(cls, data):
        with BytesIO(data) as b:
            b.read(12)
            chunk_header = b.read(4).decode('utf-8')
            if chunk_header[:3] != 'VP8':
                raise ValueError('Invalid WebP file')
            wptype = chunk_header[3]
            b.read(4)
            if wptype == 'X':
                b.read(4)
                w = int.from_bytes(b.read(3), 'little') + 1
                h = int.from_bytes(b.read(3), 'little') + 1
            elif wptype == 'L':
                b.read(1)
                bits = struct.unpack('<I', b.read(4))[0]
                w = (bits & 16383) + 1
                h = (bits >> 14 & 16383) + 1
            elif wptype == ' ':
                b.read(6)
                w = int.from_bytes(b.read(2), 'little') + 1
                h = int.from_bytes(b.read(2), 'little') + 1
        return (int(w), int(h))