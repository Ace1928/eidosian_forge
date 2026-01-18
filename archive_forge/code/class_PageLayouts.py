from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class PageLayouts:
    """Page 84, PDF 1.4 reference."""
    SINGLE_PAGE = '/SinglePage'
    ONE_COLUMN = '/OneColumn'
    TWO_COLUMN_LEFT = '/TwoColumnLeft'
    TWO_COLUMN_RIGHT = '/TwoColumnRight'