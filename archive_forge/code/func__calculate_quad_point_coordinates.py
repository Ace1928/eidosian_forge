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
def _calculate_quad_point_coordinates(x, y, width, height, angle=0):
    """
    Calculate the coordinates of rectangle when rotated by angle around x, y
    """
    angle = math.radians(-angle)
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    a = x + height * sin_angle
    b = y + height * cos_angle
    c = x + width * cos_angle + height * sin_angle
    d = y - width * sin_angle + height * cos_angle
    e = x + width * cos_angle
    f = y - width * sin_angle
    return ((x, y), (e, f), (c, d), (a, b))