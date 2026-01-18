import functools
import io
import logging
import os
import sys
import time
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from oslo_utils import timeutils
class CausedByException(Exception):
    """Base class for exceptions which have associated causes.

    NOTE(harlowja): in later versions of python we can likely remove the need
    to have a ``cause`` here as PY3+ have implemented :pep:`3134` which
    handles chaining in a much more elegant manner.

    :param message: the exception message, typically some string that is
                    useful for consumers to view when debugging or analyzing
                    failures.
    :param cause: the cause of the exception being raised, when provided this
                  should itself be an exception instance, this is useful for
                  creating a chain of exceptions for versions of python where
                  this is not yet implemented/supported natively.

    .. versionadded:: 2.4
    """

    def __init__(self, message, cause=None):
        super(CausedByException, self).__init__(message)
        self.cause = cause

    def __bytes__(self):
        return self.pformat().encode('utf8')

    def __str__(self):
        return self.pformat()

    def _get_message(self):
        return self.args[0]

    def pformat(self, indent=2, indent_text=' ', show_root_class=False):
        """Pretty formats a caused exception + any connected causes."""
        if indent < 0:
            raise ValueError("Provided 'indent' must be greater than or equal to zero instead of %s" % indent)
        buf = io.StringIO()
        if show_root_class:
            buf.write(reflection.get_class_name(self, fully_qualified=False))
            buf.write(': ')
        buf.write(self._get_message())
        active_indent = indent
        next_up = self.cause
        seen = []
        while next_up is not None and next_up not in seen:
            seen.append(next_up)
            buf.write(os.linesep)
            if isinstance(next_up, CausedByException):
                buf.write(indent_text * active_indent)
                buf.write(reflection.get_class_name(next_up, fully_qualified=False))
                buf.write(': ')
                buf.write(next_up._get_message())
            else:
                lines = traceback.format_exception_only(type(next_up), next_up)
                for i, line in enumerate(lines):
                    buf.write(indent_text * active_indent)
                    if line.endswith('\n'):
                        line = line[0:-1]
                    buf.write(line)
                    if i + 1 != len(lines):
                        buf.write(os.linesep)
                break
            active_indent += indent
            next_up = getattr(next_up, 'cause', None)
        return buf.getvalue()