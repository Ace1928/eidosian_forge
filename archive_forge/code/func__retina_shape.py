from binascii import b2a_base64, hexlify
import html
import json
import mimetypes
import os
import struct
import warnings
from copy import deepcopy
from os.path import splitext
from pathlib import Path, PurePath
from IPython.utils.py3compat import cast_unicode
from IPython.testing.skipdoctest import skip_doctest
from . import display_functions
from warnings import warn
def _retina_shape(self):
    """load pixel-doubled width and height from image data"""
    if not self.embed:
        return
    if self.format == self._FMT_PNG:
        w, h = _pngxy(self.data)
    elif self.format == self._FMT_JPEG:
        w, h = _jpegxy(self.data)
    elif self.format == self._FMT_GIF:
        w, h = _gifxy(self.data)
    else:
        return
    self.width = w // 2
    self.height = h // 2