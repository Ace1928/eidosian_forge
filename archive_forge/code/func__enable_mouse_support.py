from __future__ import annotations
import asyncio
from textual._xterm_parser import XTermParser
from textual.app import App
from textual.driver import Driver
from textual.events import Resize
from textual.geometry import Size
def _enable_mouse_support(self) -> None:
    """Enable reporting of mouse events."""
    write = self.write
    write('\x1b[?1000h')
    write('\x1b[?1003h')
    write('\x1b[?1015h')
    write('\x1b[?1006h')
    self.flush()