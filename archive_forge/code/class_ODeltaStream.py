from gitdb.util import bin_to_hex
from gitdb.fun import (
class ODeltaStream(OStream):
    """Uses size info of its stream, delaying reads"""

    def __new__(cls, sha, type, size, stream, *args, **kwargs):
        """Helps with the initialization of subclasses"""
        return tuple.__new__(cls, (sha, type, size, stream))

    @property
    def size(self):
        return self[3].size