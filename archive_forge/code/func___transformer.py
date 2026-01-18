from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def __transformer(self, box, image, method, data, resample=Resampling.NEAREST, fill=1):
    w = box[2] - box[0]
    h = box[3] - box[1]
    if method == Transform.AFFINE:
        data = data[:6]
    elif method == Transform.EXTENT:
        x0, y0, x1, y1 = data
        xs = (x1 - x0) / w
        ys = (y1 - y0) / h
        method = Transform.AFFINE
        data = (xs, 0, x0, 0, ys, y0)
    elif method == Transform.PERSPECTIVE:
        data = data[:8]
    elif method == Transform.QUAD:
        nw = data[:2]
        sw = data[2:4]
        se = data[4:6]
        ne = data[6:8]
        x0, y0 = nw
        As = 1.0 / w
        At = 1.0 / h
        data = (x0, (ne[0] - x0) * As, (sw[0] - x0) * At, (se[0] - sw[0] - ne[0] + x0) * As * At, y0, (ne[1] - y0) * As, (sw[1] - y0) * At, (se[1] - sw[1] - ne[1] + y0) * As * At)
    else:
        msg = 'unknown transformation method'
        raise ValueError(msg)
    if resample not in (Resampling.NEAREST, Resampling.BILINEAR, Resampling.BICUBIC):
        if resample in (Resampling.BOX, Resampling.HAMMING, Resampling.LANCZOS):
            msg = {Resampling.BOX: 'Image.Resampling.BOX', Resampling.HAMMING: 'Image.Resampling.HAMMING', Resampling.LANCZOS: 'Image.Resampling.LANCZOS'}[resample] + f' ({resample}) cannot be used.'
        else:
            msg = f'Unknown resampling filter ({resample}).'
        filters = [f'{filter[1]} ({filter[0]})' for filter in ((Resampling.NEAREST, 'Image.Resampling.NEAREST'), (Resampling.BILINEAR, 'Image.Resampling.BILINEAR'), (Resampling.BICUBIC, 'Image.Resampling.BICUBIC'))]
        msg += ' Use ' + ', '.join(filters[:-1]) + ' or ' + filters[-1]
        raise ValueError(msg)
    image.load()
    self.load()
    if image.mode in ('1', 'P'):
        resample = Resampling.NEAREST
    self.im.transform2(box, image.im, method, data, resample, fill)