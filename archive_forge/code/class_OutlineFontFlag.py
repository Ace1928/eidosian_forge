from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class OutlineFontFlag(IntFlag):
    """A class used as an enumerable flag for formatting an outline font."""
    italic = 1
    bold = 2