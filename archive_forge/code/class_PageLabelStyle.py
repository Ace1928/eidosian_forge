from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class PageLabelStyle:
    """Table 8.10 in the 1.7 reference."""
    DECIMAL = '/D'
    LOWERCASE_ROMAN = '/r'
    UPPERCASE_ROMAN = '/R'
    LOWERCASE_LETTER = '/a'
    UPPERCASE_LETTER = '/A'