import os
from glob import glob
import re
from collections.abc import Sequence
from copy import copy
import numpy as np
from PIL import Image
from tifffile import TiffFile
def _is_multipattern(input_pattern):
    """Helping function. Returns True if pattern contains a tuple, list, or a
    string separated with os.pathsep."""
    has_str_ospathsep = isinstance(input_pattern, str) and os.pathsep in input_pattern
    not_a_string = not isinstance(input_pattern, str)
    has_iterable = isinstance(input_pattern, Sequence)
    has_strings = all((isinstance(pat, str) for pat in input_pattern))
    is_multipattern = has_str_ospathsep or (not_a_string and has_iterable and has_strings)
    return is_multipattern