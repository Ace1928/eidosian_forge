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
class GIF(ImageBase):
    """
    The `GIF` pane embeds a .gif image file in a panel if provided a local
    path, or will link to a remote image if provided a URL.

    Reference: https://panel.holoviz.org/reference/panes/GIF.html

    :Example:

    >>> GIF(
    ...     'https://upload.wikimedia.org/wikipedia/commons/b/b1/Loading_icon.gif',
    ...     alt_text='A loading spinner',
    ...     link_url='https://commons.wikimedia.org/wiki/File:Loading_icon.gif',
    ...     width=500
    ... )
    """
    filetype: ClassVar[str] = 'gif'

    @classmethod
    def _imgshape(cls, data):
        import struct
        w, h = struct.unpack('<HH', data[6:10])
        return (int(w), int(h))