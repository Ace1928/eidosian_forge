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
def _jpegxy(data):
    """read the (width, height) from a JPEG header"""
    idx = 4
    while True:
        block_size = struct.unpack('>H', data[idx:idx + 2])[0]
        idx = idx + block_size
        if data[idx:idx + 2] == b'\xff\xc0':
            iSOF = idx
            break
        else:
            idx += 2
    h, w = struct.unpack('>HH', data[iSOF + 5:iSOF + 9])
    return (w, h)