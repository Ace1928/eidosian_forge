from __future__ import annotations
import asyncio
from textual._xterm_parser import XTermParser
from textual.app import App
from textual.driver import Driver
from textual.events import Resize
from textual.geometry import Size
def _disable_mouse_support(self) -> None:
    """Disable reporting of mouse events."""
    write = self.write
    write('\x1b[?1000l')
    write('\x1b[?1003l')
    write('\x1b[?1015l')
    write('\x1b[?1006l')
    self.flush()