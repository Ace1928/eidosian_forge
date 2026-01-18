from hashlib import sha1
from ..util import compat
from ..util import langhelpers
class repr_obj:
    __slots__ = ('value', 'max_chars')

    def __init__(self, value, max_chars=300):
        self.value = value
        self.max_chars = max_chars

    def __eq__(self, other):
        return other.value == self.value

    def __repr__(self):
        rep = repr(self.value)
        lenrep = len(rep)
        if lenrep > self.max_chars:
            segment_length = self.max_chars // 2
            rep = rep[0:segment_length] + ' ... (%d characters truncated) ... ' % (lenrep - self.max_chars) + rep[-segment_length:]
        return rep