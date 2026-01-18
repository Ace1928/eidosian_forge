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
class DebugFileReporterWrapper(FileReporter):
    """A debugging `FileReporter`."""

    def __init__(self, filename: str, reporter: FileReporter, debug: LabelledDebug) -> None:
        super().__init__(filename)
        self.reporter = reporter
        self.debug = debug

    def relative_filename(self) -> str:
        ret = self.reporter.relative_filename()
        self.debug.write(f'relative_filename() --> {ret!r}')
        return ret

    def lines(self) -> set[TLineNo]:
        ret = self.reporter.lines()
        self.debug.write(f'lines() --> {ret!r}')
        return ret

    def excluded_lines(self) -> set[TLineNo]:
        ret = self.reporter.excluded_lines()
        self.debug.write(f'excluded_lines() --> {ret!r}')
        return ret

    def translate_lines(self, lines: Iterable[TLineNo]) -> set[TLineNo]:
        ret = self.reporter.translate_lines(lines)
        self.debug.write(f'translate_lines({lines!r}) --> {ret!r}')
        return ret

    def translate_arcs(self, arcs: Iterable[TArc]) -> set[TArc]:
        ret = self.reporter.translate_arcs(arcs)
        self.debug.write(f'translate_arcs({arcs!r}) --> {ret!r}')
        return ret

    def no_branch_lines(self) -> set[TLineNo]:
        ret = self.reporter.no_branch_lines()
        self.debug.write(f'no_branch_lines() --> {ret!r}')
        return ret

    def exit_counts(self) -> dict[TLineNo, int]:
        ret = self.reporter.exit_counts()
        self.debug.write(f'exit_counts() --> {ret!r}')
        return ret

    def arcs(self) -> set[TArc]:
        ret = self.reporter.arcs()
        self.debug.write(f'arcs() --> {ret!r}')
        return ret

    def source(self) -> str:
        ret = self.reporter.source()
        self.debug.write('source() --> %d chars' % (len(ret),))
        return ret

    def source_token_lines(self) -> TSourceTokenLines:
        ret = list(self.reporter.source_token_lines())
        self.debug.write('source_token_lines() --> %d tokens' % (len(ret),))
        return ret