from __future__ import annotations
import itertools
import math
import os
import subprocess
from enum import IntEnum
from . import (
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
def _write_single_frame(im, fp, palette):
    im_out = _normalize_mode(im)
    for k, v in im_out.info.items():
        im.encoderinfo.setdefault(k, v)
    im_out = _normalize_palette(im_out, palette, im.encoderinfo)
    for s in _get_global_header(im_out, im.encoderinfo):
        fp.write(s)
    flags = 0
    if get_interlace(im):
        flags = flags | 64
    _write_local_header(fp, im, (0, 0), flags)
    im_out.encoderconfig = (8, get_interlace(im))
    ImageFile._save(im_out, fp, [('gif', (0, 0) + im.size, 0, RAWMODE[im_out.mode])])
    fp.write(b'\x00')