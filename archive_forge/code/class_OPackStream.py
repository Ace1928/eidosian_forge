from gitdb.util import bin_to_hex
from gitdb.fun import (
class OPackStream(OPackInfo):
    """Next to pack object information, a stream outputting an undeltified base object
    is provided"""
    __slots__ = tuple()

    def __new__(cls, packoffset, type, size, stream, *args):
        """Helps with the initialization of subclasses"""
        return tuple.__new__(cls, (packoffset, type, size, stream))

    def read(self, size=-1):
        return self[3].read(size)

    @property
    def stream(self):
        return self[3]