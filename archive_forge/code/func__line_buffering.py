import contextlib
import enum
import io
import os
import signal
import subprocess
import sys
import types
import typing
from typing import Any, Optional, Type, Dict, TextIO
from autopage import command
def _line_buffering(self) -> bool:
    if self._set_line_buffering is None:
        return getattr(self._out, 'line_buffering', self._tty)
    return self._set_line_buffering