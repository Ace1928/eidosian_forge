from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _set_msword_style(self) -> None:
    self.header = True
    self.border = True
    self._hrules = NONE
    self.padding_width = 1
    self.left_padding_width = 1
    self.right_padding_width = 1
    self.vertical_char = '|'