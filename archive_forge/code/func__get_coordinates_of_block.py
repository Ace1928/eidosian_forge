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
def _get_coordinates_of_block(x, y, width, height, angle=0):
    """
    Get the coordinates of rotated rectangle and rectangle that covers the
    rotated rectangle.
    """
    vertices = _calculate_quad_point_coordinates(x, y, width, height, angle)
    pad = 1e-05 if angle % 90 else 0
    min_x = min((v[0] for v in vertices)) - pad
    min_y = min((v[1] for v in vertices)) - pad
    max_x = max((v[0] for v in vertices)) + pad
    max_y = max((v[1] for v in vertices)) + pad
    return (tuple(itertools.chain.from_iterable(vertices)), (min_x, min_y, max_x, max_y))