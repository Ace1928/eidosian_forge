from typing import TYPE_CHECKING, Optional
from .align import AlignMethod
from .box import ROUNDED, Box
from .cells import cell_len
from .jupyter import JupyterMixin
from .measure import Measurement, measure_renderables
from .padding import Padding, PaddingDimensions
from .segment import Segment
from .style import Style, StyleType
from .text import Text, TextType
@property
def _subtitle(self) -> Optional[Text]:
    if self.subtitle:
        subtitle_text = Text.from_markup(self.subtitle) if isinstance(self.subtitle, str) else self.subtitle.copy()
        subtitle_text.end = ''
        subtitle_text.plain = subtitle_text.plain.replace('\n', ' ')
        subtitle_text.no_wrap = True
        subtitle_text.expand_tabs()
        subtitle_text.pad(1)
        return subtitle_text
    return None