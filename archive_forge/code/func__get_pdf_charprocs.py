import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def _get_pdf_charprocs(font_path, glyph_ids):
    font = get_font(font_path, hinting_factor=1)
    conv = 1000 / font.units_per_EM
    procs = {}
    for glyph_id in glyph_ids:
        g = font.load_glyph(glyph_id, LOAD_NO_SCALE)
        d1 = (np.array([g.horiAdvance, 0, *g.bbox]) * conv + 0.5).astype(int)
        v, c = font.get_path()
        v = (v * 64).astype(int)
        quads, = np.nonzero(c == 3)
        quads_on = quads[1::2]
        quads_mid_on = np.array(sorted({*quads_on} & {*quads - 1} & {*quads + 1}), int)
        implicit = quads_mid_on[(v[quads_mid_on] == ((v[quads_mid_on - 1] + v[quads_mid_on + 1]) / 2).astype(int)).all(axis=1)]
        if (font.postscript_name, glyph_id) in [('DejaVuSerif-Italic', 77), ('DejaVuSerif-Italic', 135)]:
            v[:, 0] -= 1
        v = (v * conv + 0.5).astype(int)
        v[implicit] = ((v[implicit - 1] + v[implicit + 1]) / 2).astype(int)
        procs[font.get_glyph_name(glyph_id)] = ' '.join(map(str, d1)).encode('ascii') + b' d1\n' + _path.convert_to_string(Path(v, c), None, None, False, None, -1, [b'm', b'l', b'', b'c', b'h'], True) + b'f'
    return procs