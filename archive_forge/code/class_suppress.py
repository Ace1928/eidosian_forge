import sys
from contextlib import AbstractContextManager
class suppress(AbstractContextManager):
    """Backport of :class:`contextlib.suppress` from Python 3.12.1."""

    def __init__(self, *exceptions):
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        if exctype is None:
            return
        if issubclass(exctype, self._exceptions):
            return True
        if issubclass(exctype, BaseExceptionGroup):
            match, rest = excinst.split(self._exceptions)
            if rest is None:
                return True
            raise rest
        return False