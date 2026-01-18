import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase
def _write_clips(self):
    if not len(self._clipd):
        return
    writer = self.writer
    writer.start('defs')
    for clip, oid in self._clipd.values():
        writer.start('clipPath', id=oid)
        if len(clip) == 2:
            clippath, clippath_trans = clip
            path_data = self._convert_path(clippath, clippath_trans, simplify=False)
            writer.element('path', d=path_data)
        else:
            x, y, w, h = clip
            writer.element('rect', x=_short_float_fmt(x), y=_short_float_fmt(y), width=_short_float_fmt(w), height=_short_float_fmt(h))
        writer.end('clipPath')
    writer.end('defs')