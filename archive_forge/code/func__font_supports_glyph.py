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
def _font_supports_glyph(fonttype, glyph):
    """
    Returns True if the font is able to provide codepoint *glyph* in a PDF.

    For a Type 3 font, this method returns True only for single-byte
    characters. For Type 42 fonts this method return True if the character is
    from the Basic Multilingual Plane.
    """
    if fonttype == 3:
        return glyph <= 255
    if fonttype == 42:
        return glyph <= 65535
    raise NotImplementedError()