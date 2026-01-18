from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
class LCDScreen(BaseScreen, abc.ABC):
    """Base class for LCD-based screens."""
    DISPLAY_SIZE: tuple[int, int]

    def set_terminal_properties(self, colors: Literal[1, 16, 88, 256, 16777216] | None=None, bright_is_bold: bool | None=None, has_underline: bool | None=None) -> None:
        pass

    def set_input_timeouts(self, *args):
        pass

    def reset_default_terminal_palette(self, *args):
        pass

    def get_cols_rows(self):
        return self.DISPLAY_SIZE