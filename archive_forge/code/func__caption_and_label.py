from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _caption_and_label(self) -> str:
    if self.caption or self.label:
        double_backslash = '\\\\'
        elements = [f'{self._caption_macro}', f'{self._label_macro}']
        caption_and_label = '\n'.join([item for item in elements if item])
        caption_and_label += double_backslash
        return caption_and_label
    else:
        return ''