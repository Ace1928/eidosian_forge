from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
@staticmethod
def _get_normalization_factor(max_abs_value, normalize):
    if not normalize and max_abs_value > 1:
        raise ValueError('Audio data must be between -1 and 1 when normalize=False.')
    return max_abs_value if normalize else 1