import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def _alpha_float(self) -> float:
    return 1 if self._rgba.alpha is None else self._rgba.alpha