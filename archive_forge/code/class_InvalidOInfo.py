from gitdb.util import bin_to_hex
from gitdb.fun import (
class InvalidOInfo(tuple):
    """Carries information about a sha identifying an object which is invalid in
    the queried database. The exception attribute provides more information about
    the cause of the issue"""
    __slots__ = tuple()

    def __new__(cls, sha, exc):
        return tuple.__new__(cls, (sha, exc))

    def __init__(self, sha, exc):
        tuple.__init__(self, (sha, exc))

    @property
    def binsha(self):
        return self[0]

    @property
    def hexsha(self):
        return bin_to_hex(self[0])

    @property
    def error(self):
        """:return: exception instance explaining the failure"""
        return self[1]