from __future__ import annotations
import tkinter
from io import BytesIO
from . import Image
def _get_image_from_kw(kw):
    source = None
    if 'file' in kw:
        source = kw.pop('file')
    elif 'data' in kw:
        source = BytesIO(kw.pop('data'))
    if source:
        return Image.open(source)