from __future__ import annotations
import asyncio
from textual._xterm_parser import XTermParser
from textual.app import App
from textual.driver import Driver
from textual.events import Resize
from textual.geometry import Size
def _disable_bracketed_paste(self) -> None:
    """Disable bracketed paste mode."""
    self.write('\x1b[?2004l')