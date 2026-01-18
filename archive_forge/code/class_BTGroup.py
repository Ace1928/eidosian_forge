import sys
from itertools import groupby
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from ..._utils import logger_warning
from .. import LAYOUT_NEW_BT_GROUP_SPACE_WIDTHS
from ._font import Font
from ._text_state_manager import TextStateManager
from ._text_state_params import TextStateParams
class BTGroup(TypedDict):
    """
    Dict describing a line of text rendered within a BT/ET operator pair.
    If multiple text show operations render text on the same line, the text
    will be combined into a single BTGroup dict.

    Keys:
        tx: x coordinate of first character in BTGroup
        ty: y coordinate of first character in BTGroup
        font_size: nominal font size
        font_height: effective font height
        text: rendered text
        displaced_tx: x coordinate of last character in BTGroup
        flip_sort: -1 if page is upside down, else 1
    """
    tx: float
    ty: float
    font_size: float
    font_height: float
    text: str
    displaced_tx: float
    flip_sort: Literal[-1, 1]