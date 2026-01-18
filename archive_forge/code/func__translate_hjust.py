from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from .element_base import element_base
from .margin import Margin
def _translate_hjust(self, just: float) -> Literal['left', 'right', 'center']:
    """
        Translate ggplot2 justification from [0, 1] to left, right, center.
        """
    if just == 0:
        return 'left'
    elif just == 1:
        return 'right'
    else:
        return 'center'