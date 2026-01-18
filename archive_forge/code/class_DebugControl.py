from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
class DebugControl:
    """Control and output for debugging."""
    show_repr_attr = False

    def __init__(self, options: Iterable[str], output: IO[str] | None, file_name: str | None=None) -> None:
        """Configure the options and output file for debugging."""
        self.options = list(options) + FORCED_DEBUG
        self.suppress_callers = False
        filters = []
        if self.should('process'):
            filters.append(CwdTracker().filter)
            filters.append(ProcessTracker().filter)
        if self.should('pytest'):
            filters.append(PytestTracker().filter)
        if self.should('pid'):
            filters.append(add_pid_and_tid)
        self.output = DebugOutputFile.get_one(output, file_name=file_name, filters=filters)
        self.raw_output = self.output.outfile

    def __repr__(self) -> str:
        return f'<DebugControl options={self.options!r} raw_output={self.raw_output!r}>'

    def should(self, option: str) -> bool:
        """Decide whether to output debug information in category `option`."""
        if option == 'callers' and self.suppress_callers:
            return False
        return option in self.options

    @contextlib.contextmanager
    def without_callers(self) -> Iterator[None]:
        """A context manager to prevent call stacks from being logged."""
        old = self.suppress_callers
        self.suppress_callers = True
        try:
            yield
        finally:
            self.suppress_callers = old

    def write(self, msg: str, *, exc: BaseException | None=None) -> None:
        """Write a line of debug output.

        `msg` is the line to write. A newline will be appended.

        If `exc` is provided, a stack trace of the exception will be written
        after the message.

        """
        self.output.write(msg + '\n')
        if exc is not None:
            self.output.write(''.join(traceback.format_exception(None, exc, exc.__traceback__)))
        if self.should('self'):
            caller_self = inspect.stack()[1][0].f_locals.get('self')
            if caller_self is not None:
                self.output.write(f'self: {caller_self!r}\n')
        if self.should('callers'):
            dump_stack_frames(out=self.output, skip=1)
        self.output.flush()