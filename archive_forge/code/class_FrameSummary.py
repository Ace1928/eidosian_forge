import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
class FrameSummary:
    """Information about a single frame from a traceback.

    - :attr:`filename` The filename for the frame.
    - :attr:`lineno` The line within filename for the frame that was
      active when the frame was captured.
    - :attr:`name` The name of the function or method that was executing
      when the frame was captured.
    - :attr:`line` The text from the linecache module for the
      of code that was running when the frame was captured.
    - :attr:`locals` Either None if locals were not supplied, or a dict
      mapping the name to the repr() of the variable.
    """
    __slots__ = ('filename', 'lineno', 'end_lineno', 'colno', 'end_colno', 'name', '_line', 'locals')

    def __init__(self, filename, lineno, name, *, lookup_line=True, locals=None, line=None, end_lineno=None, colno=None, end_colno=None):
        """Construct a FrameSummary.

        :param lookup_line: If True, `linecache` is consulted for the source
            code line. Otherwise, the line will be looked up when first needed.
        :param locals: If supplied the frame locals, which will be captured as
            object representations.
        :param line: If provided, use this instead of looking up the line in
            the linecache.
        """
        self.filename = filename
        self.lineno = lineno
        self.name = name
        self._line = line
        if lookup_line:
            self.line
        self.locals = {k: repr(v) for k, v in locals.items()} if locals else None
        self.end_lineno = end_lineno
        self.colno = colno
        self.end_colno = end_colno

    def __eq__(self, other):
        if isinstance(other, FrameSummary):
            return self.filename == other.filename and self.lineno == other.lineno and (self.name == other.name) and (self.locals == other.locals)
        if isinstance(other, tuple):
            return (self.filename, self.lineno, self.name, self.line) == other
        return NotImplemented

    def __getitem__(self, pos):
        return (self.filename, self.lineno, self.name, self.line)[pos]

    def __iter__(self):
        return iter([self.filename, self.lineno, self.name, self.line])

    def __repr__(self):
        return '<FrameSummary file {filename}, line {lineno} in {name}>'.format(filename=self.filename, lineno=self.lineno, name=self.name)

    def __len__(self):
        return 4

    @property
    def _original_line(self):
        self.line
        return self._line

    @property
    def line(self):
        if self._line is None:
            if self.lineno is None:
                return None
            self._line = linecache.getline(self.filename, self.lineno)
        return self._line.strip()