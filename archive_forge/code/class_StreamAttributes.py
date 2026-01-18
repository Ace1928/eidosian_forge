from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class StreamAttributes:
    """Table 4.2."""
    LENGTH = '/Length'
    FILTER = '/Filter'
    DECODE_PARMS = '/DecodeParms'