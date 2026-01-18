from __future__ import annotations
import logging
import re
import sys
import typing as t
from datetime import datetime
from datetime import timezone
class _ColorStreamHandler(logging.StreamHandler):
    """On Windows, wrap stream with Colorama for ANSI style support."""

    def __init__(self) -> None:
        try:
            import colorama
        except ImportError:
            stream = None
        else:
            stream = colorama.AnsiToWin32(sys.stderr)
        super().__init__(stream)