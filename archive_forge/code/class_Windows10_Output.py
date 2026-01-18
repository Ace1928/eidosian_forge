from __future__ import annotations
import sys
from ctypes import byref, windll
from ctypes.wintypes import DWORD, HANDLE
from typing import Any, TextIO
from prompt_toolkit.data_structures import Size
from prompt_toolkit.win32_types import STD_OUTPUT_HANDLE
from .base import Output
from .color_depth import ColorDepth
from .vt100 import Vt100_Output
from .win32 import Win32Output
class Windows10_Output:
    """
    Windows 10 output abstraction. This enables and uses vt100 escape sequences.
    """

    def __init__(self, stdout: TextIO, default_color_depth: ColorDepth | None=None) -> None:
        self.default_color_depth = default_color_depth
        self.win32_output = Win32Output(stdout, default_color_depth=default_color_depth)
        self.vt100_output = Vt100_Output(stdout, lambda: Size(0, 0), default_color_depth=default_color_depth)
        self._hconsole = HANDLE(windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE))

    def flush(self) -> None:
        """
        Write to output stream and flush.
        """
        original_mode = DWORD(0)
        windll.kernel32.GetConsoleMode(self._hconsole, byref(original_mode))
        windll.kernel32.SetConsoleMode(self._hconsole, DWORD(ENABLE_PROCESSED_INPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        try:
            self.vt100_output.flush()
        finally:
            windll.kernel32.SetConsoleMode(self._hconsole, original_mode)

    @property
    def responds_to_cpr(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        if name in ('get_size', 'get_rows_below_cursor_position', 'enable_mouse_support', 'disable_mouse_support', 'scroll_buffer_to_prompt', 'get_win32_screen_buffer_info', 'enable_bracketed_paste', 'disable_bracketed_paste'):
            return getattr(self.win32_output, name)
        else:
            return getattr(self.vt100_output, name)

    def get_default_color_depth(self) -> ColorDepth:
        """
        Return the default color depth for a windows terminal.

        Contrary to the Vt100 implementation, this doesn't depend on a $TERM
        variable.
        """
        if self.default_color_depth is not None:
            return self.default_color_depth
        return ColorDepth.TRUE_COLOR