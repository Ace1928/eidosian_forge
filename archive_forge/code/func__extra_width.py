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
@property
def _extra_width(self) -> int:
    """Get extra width to add to cell content."""
    width = 0
    if self.box and self.show_edge:
        width += 2
    if self.box:
        width += len(self.columns) - 1
    return width