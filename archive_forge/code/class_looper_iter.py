import sys
from Cython.Tempita.compat3 import basestring_
class looper_iter(object):

    def __init__(self, seq):
        self.seq = list(seq)
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.seq):
            raise StopIteration
        result = (loop_pos(self.seq, self.pos), self.seq[self.pos])
        self.pos += 1
        return result
    if sys.version < '3':
        next = __next__