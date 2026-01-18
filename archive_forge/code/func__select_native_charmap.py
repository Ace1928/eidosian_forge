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
@staticmethod
def _select_native_charmap(font):
    for charmap_code in [1094992451, 1094995778]:
        try:
            font.select_charmap(charmap_code)
        except (ValueError, RuntimeError):
            pass
        else:
            break
    else:
        _log.warning('No supported encoding in font (%s).', font.fname)