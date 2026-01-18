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
def _get_link_annotation(gc, x, y, width, height, angle=0):
    """
    Create a link annotation object for embedding URLs.
    """
    quadpoints, rect = _get_coordinates_of_block(x, y, width, height, angle)
    link_annotation = {'Type': Name('Annot'), 'Subtype': Name('Link'), 'Rect': rect, 'Border': [0, 0, 0], 'A': {'S': Name('URI'), 'URI': gc.get_url()}}
    if angle % 90:
        link_annotation['QuadPoints'] = quadpoints
    return link_annotation