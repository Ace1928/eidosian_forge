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
class DebugPluginWrapper(CoveragePlugin):
    """Wrap a plugin, and use debug to report on what it's doing."""

    def __init__(self, plugin: CoveragePlugin, debug: LabelledDebug) -> None:
        super().__init__()
        self.plugin = plugin
        self.debug = debug

    def file_tracer(self, filename: str) -> FileTracer | None:
        tracer = self.plugin.file_tracer(filename)
        self.debug.write(f'file_tracer({filename!r}) --> {tracer!r}')
        if tracer:
            debug = self.debug.add_label(f'file {filename!r}')
            tracer = DebugFileTracerWrapper(tracer, debug)
        return tracer

    def file_reporter(self, filename: str) -> FileReporter | str:
        reporter = self.plugin.file_reporter(filename)
        assert isinstance(reporter, FileReporter)
        self.debug.write(f'file_reporter({filename!r}) --> {reporter!r}')
        if reporter:
            debug = self.debug.add_label(f'file {filename!r}')
            reporter = DebugFileReporterWrapper(filename, reporter, debug)
        return reporter

    def dynamic_context(self, frame: FrameType) -> str | None:
        context = self.plugin.dynamic_context(frame)
        self.debug.write(f'dynamic_context({frame!r}) --> {context!r}')
        return context

    def find_executable_files(self, src_dir: str) -> Iterable[str]:
        executable_files = self.plugin.find_executable_files(src_dir)
        self.debug.write(f'find_executable_files({src_dir!r}) --> {executable_files!r}')
        return executable_files

    def configure(self, config: TConfigurable) -> None:
        self.debug.write(f'configure({config!r})')
        self.plugin.configure(config)

    def sys_info(self) -> Iterable[tuple[str, Any]]:
        return self.plugin.sys_info()