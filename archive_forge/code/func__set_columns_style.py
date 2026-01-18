from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _set_columns_style(self) -> None:
    self.header = True
    self.border = False
    self.padding_width = 1
    self.left_padding_width = 0
    self.right_padding_width = 8