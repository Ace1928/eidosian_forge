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
class DebugFileTracerWrapper(FileTracer):
    """A debugging `FileTracer`."""

    def __init__(self, tracer: FileTracer, debug: LabelledDebug) -> None:
        self.tracer = tracer
        self.debug = debug

    def _show_frame(self, frame: FrameType) -> str:
        """A short string identifying a frame, for debug messages."""
        return '%s@%d' % (os.path.basename(frame.f_code.co_filename), frame.f_lineno)

    def source_filename(self) -> str:
        sfilename = self.tracer.source_filename()
        self.debug.write(f'source_filename() --> {sfilename!r}')
        return sfilename

    def has_dynamic_source_filename(self) -> bool:
        has = self.tracer.has_dynamic_source_filename()
        self.debug.write(f'has_dynamic_source_filename() --> {has!r}')
        return has

    def dynamic_source_filename(self, filename: str, frame: FrameType) -> str | None:
        dyn = self.tracer.dynamic_source_filename(filename, frame)
        self.debug.write('dynamic_source_filename({!r}, {}) --> {!r}'.format(filename, self._show_frame(frame), dyn))
        return dyn

    def line_number_range(self, frame: FrameType) -> tuple[TLineNo, TLineNo]:
        pair = self.tracer.line_number_range(frame)
        self.debug.write(f'line_number_range({self._show_frame(frame)}) --> {pair!r}')
        return pair