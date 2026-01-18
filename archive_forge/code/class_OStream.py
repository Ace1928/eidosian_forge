from gitdb.util import bin_to_hex
from gitdb.fun import (
class OStream(OInfo):
    """Base for object streams retrieved from the database, providing additional
    information about the stream.
    Generally, ODB streams are read-only as objects are immutable"""
    __slots__ = tuple()

    def __new__(cls, sha, type, size, stream, *args, **kwargs):
        """Helps with the initialization of subclasses"""
        return tuple.__new__(cls, (sha, type, size, stream))

    def __init__(self, *args, **kwargs):
        tuple.__init__(self)

    def read(self, size=-1):
        return self[3].read(size)

    @property
    def stream(self):
        return self[3]