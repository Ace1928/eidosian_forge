from io import StringIO
from Bio.File import _IndexedSeqFileProxy, _open_for_random_access
class SearchIndexer(_IndexedSeqFileProxy):
    """Base class for file format specific random access.

    Subclasses for each file format should define '_parser' and optionally
    'get_raw' methods.
    """

    def __init__(self, filename, **kwargs):
        """Initialize the class."""
        self._handle = _open_for_random_access(filename)
        self._kwargs = kwargs

    def _parse(self, handle):
        """Pass handle and arguments to the next iterable (PRIVATE)."""
        return next(iter(self._parser(handle, **self._kwargs)))

    def get(self, offset):
        """Get offset and convert it from bytes to string."""
        return self._parse(StringIO(self.get_raw(offset).decode()))