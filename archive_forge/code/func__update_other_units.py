from __future__ import annotations
import re
from typing import (
import warnings
from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level
def _update_other_units(self, props: dict[str, str]) -> dict[str, str]:
    font_size = self._get_font_size(props)
    for side in self.SIDES:
        prop = f'border-{side}-width'
        if prop in props:
            props[prop] = self.size_to_pt(props[prop], em_pt=font_size, conversions=self.BORDER_WIDTH_RATIOS)
        for prop in [f'margin-{side}', f'padding-{side}']:
            if prop in props:
                props[prop] = self.size_to_pt(props[prop], em_pt=font_size, conversions=self.MARGIN_RATIOS)
    return props