from gitdb.util import bin_to_hex
from gitdb.fun import (
class ODeltaPackStream(ODeltaPackInfo):
    """Provides a stream outputting the uncompressed offset delta information"""
    __slots__ = tuple()

    def __new__(cls, packoffset, type, size, delta_info, stream):
        return tuple.__new__(cls, (packoffset, type, size, delta_info, stream))

    def read(self, size=-1):
        return self[4].read(size)

    @property
    def stream(self):
        return self[4]