from __future__ import annotations
import os
import os.path
import sys
from types import FrameType
from typing import Any, Iterable, Iterator
from coverage.exceptions import PluginError
from coverage.misc import isolate_module
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter
from coverage.types import (
def _show_frame(self, frame: FrameType) -> str:
    """A short string identifying a frame, for debug messages."""
    return '%s@%d' % (os.path.basename(frame.f_code.co_filename), frame.f_lineno)