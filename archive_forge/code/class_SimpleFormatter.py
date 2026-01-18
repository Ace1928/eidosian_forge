from __future__ import annotations
from flake8.formatting import base
from flake8.violation import Violation
class SimpleFormatter(base.BaseFormatter):
    """Simple abstraction for Default and Pylint formatter commonality.

    Sub-classes of this need to define an ``error_format`` attribute in order
    to succeed. The ``format`` method relies on that attribute and expects the
    ``error_format`` string to use the old-style formatting strings with named
    parameters:

    * code
    * text
    * path
    * row
    * col

    """
    error_format: str

    def format(self, error: Violation) -> str | None:
        """Format and write error out.

        If an output filename is specified, write formatted errors to that
        file. Otherwise, print the formatted error to standard out.
        """
        return self.error_format % {'code': error.code, 'text': error.text, 'path': error.filename, 'row': error.line_number, 'col': error.column_number, **(COLORS if self.color else COLORS_OFF)}