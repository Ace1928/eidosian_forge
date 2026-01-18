from __future__ import annotations
import traceback
from typing import Any, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.i18n import _
class ExceptionChainer(BrickException):
    """A Exception that can contain a group of exceptions.

    This exception serves as a container for exceptions, useful when we want to
    store all exceptions that happened during a series of steps and then raise
    them all together as one.

    The representation of the exception will include all exceptions and their
    tracebacks.

    This class also includes a context manager for convenience, one that will
    support both swallowing the exception as if nothing had happened and
    raising the exception.  In both cases the exception will be stored.

    If a message is provided to the context manager it will be formatted and
    logged with warning level.
    """

    def __init__(self, *args, **kwargs):
        self._exceptions: list[tuple] = []
        self._repr: Optional[str] = None
        self._exc_msg_args = []
        super(ExceptionChainer, self).__init__(*args, **kwargs)

    def __repr__(self):
        if not self._repr:
            tracebacks = (''.join(traceback.format_exception(*e)).replace('\n', '\n\t') for e in self._exceptions)
            self._repr = '\n'.join(('\nChained Exception #%s\n\t%s' % (i + 1, t) for i, t in enumerate(tracebacks)))
        return self._repr
    __str__ = __repr__

    def __nonzero__(self) -> bool:
        return bool(self._exceptions)
    __bool__ = __nonzero__

    def add_exception(self, exc_type, exc_val, exc_tb) -> None:
        self._repr = None
        self._exceptions.append((exc_type, exc_val, exc_tb))

    def context(self, catch_exception: bool, msg: str='', *msg_args: Any):
        self._catch_exception = catch_exception
        self._exc_msg = msg
        self._exc_msg_args = list(msg_args)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.add_exception(exc_type, exc_val, exc_tb)
            if self._exc_msg:
                LOG.warning(self._exc_msg, *self._exc_msg_args)
            if self._catch_exception:
                return True