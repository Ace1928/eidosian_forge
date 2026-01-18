from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _set_single_border_style(self) -> None:
    self.horizontal_char = '─'
    self.vertical_char = '│'
    self.junction_char = '┼'
    self.top_junction_char = '┬'
    self.bottom_junction_char = '┴'
    self.right_junction_char = '┤'
    self.left_junction_char = '├'
    self.top_right_junction_char = '┐'
    self.top_left_junction_char = '┌'
    self.bottom_right_junction_char = '┘'
    self.bottom_left_junction_char = '└'