from collections import OrderedDict
import logging
import urllib.parse
import numpy as np
from matplotlib import _text_helpers, dviread
from matplotlib.font_manager import (
from matplotlib.ft2font import LOAD_NO_HINTING, LOAD_TARGET_LIGHT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
def _get_char_id(self, font, ccode):
    """
        Return a unique id for the given font and character-code set.
        """
    return urllib.parse.quote(f'{font.postscript_name}-{ccode:x}')