from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class LzwFilterParameters:
    """Table 4.4."""
    PREDICTOR = '/Predictor'
    COLUMNS = '/Columns'
    COLORS = '/Colors'
    BITS_PER_COMPONENT = '/BitsPerComponent'
    EARLY_CHANGE = '/EarlyChange'