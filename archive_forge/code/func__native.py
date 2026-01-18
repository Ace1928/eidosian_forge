from contextlib import suppress
from io import TextIOWrapper
from . import abc
def _native(self):
    """
        Return the native reader if it supports files().
        """
    reader = self._reader
    return reader if hasattr(reader, 'files') else self