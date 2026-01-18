import io
import os
import sys
from pyasn1 import error
from pyasn1.type import univ
class CachingStreamWrapper(io.IOBase):
    """Wrapper around non-seekable streams.

    Note that the implementation is tied to the decoder,
    not checking for dangerous arguments for the sake
    of performance.

    The read bytes are kept in an internal cache until
    setting _markedPosition which may reset the cache.
    """

    def __init__(self, raw):
        self._raw = raw
        self._cache = io.BytesIO()
        self._markedPosition = 0

    def peek(self, n):
        result = self.read(n)
        self._cache.seek(-len(result), os.SEEK_CUR)
        return result

    def seekable(self):
        return True

    def seek(self, n=-1, whence=os.SEEK_SET):
        return self._cache.seek(n, whence)

    def read(self, n=-1):
        read_from_cache = self._cache.read(n)
        if n != -1:
            n -= len(read_from_cache)
            if not n:
                return read_from_cache
        read_from_raw = self._raw.read(n)
        self._cache.write(read_from_raw)
        return read_from_cache + read_from_raw

    @property
    def markedPosition(self):
        """Position where the currently processed element starts.

        This is used for back-tracking in SingleItemDecoder.__call__
        and (indefLen)ValueDecoder and should not be used for other purposes.
        The client is not supposed to ever seek before this position.
        """
        return self._markedPosition

    @markedPosition.setter
    def markedPosition(self, value):
        self._markedPosition = value
        if self._cache.tell() > io.DEFAULT_BUFFER_SIZE:
            self._cache = io.BytesIO(self._cache.read())
            self._markedPosition = 0

    def tell(self):
        return self._cache.tell()