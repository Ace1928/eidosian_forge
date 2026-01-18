from dataclasses import dataclass, field, replace
from typing import (
from . import box, errors
from ._loop import loop_first_last, loop_last
from ._pick import pick_bool
from ._ratio import ratio_distribute, ratio_reduce
from .align import VerticalAlignMethod
from .jupyter import JupyterMixin
from .measure import Measurement
from .padding import Padding, PaddingDimensions
from .protocol import is_renderable
from .segment import Segment
from .style import Style, StyleType
from .text import Text, TextType
def _get_padding_width(self, column_index: int) -> int:
    """Get extra width from padding."""
    _, pad_right, _, pad_left = self.padding
    if self.collapse_padding:
        if column_index > 0:
            pad_left = max(0, pad_left - pad_right)
    return pad_left + pad_right