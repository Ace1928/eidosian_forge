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
def _img_dims(self, width, height):
    smode = self.sizing_mode
    if smode in ['fixed', None]:
        w, h = (f'{width}px' if width else 'auto', f'{height}px' if height else 'auto')
    elif smode == 'stretch_both' and (not self.fixed_aspect):
        w, h = ('100%', '100%')
    elif smode == 'stretch_width' and (not self.fixed_aspect):
        w, h = ('100%', f'{height}px' if height else 'auto')
    elif smode == 'stretch_height' and (not self.fixed_aspect):
        w, h = (f'{width}px' if width else 'auto', '100%')
    elif smode in ('scale_height', 'stretch_height'):
        w, h = ('auto', '100%')
    else:
        w, h = ('100%', 'auto')
    return (w, h)