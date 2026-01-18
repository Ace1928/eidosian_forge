import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
class _Accumulator(list):

    def __init__(self, initial_size=0):
        self._initial_size = initial_size
        super().__init__()

    def push(self, fws, string):
        self.append((fws, string))

    def pop_from(self, i=0):
        popped = self[i:]
        self[i:] = []
        return popped

    def pop(self):
        if self.part_count() == 0:
            return ('', '')
        return super().pop()

    def __len__(self):
        return sum((len(fws) + len(part) for fws, part in self), self._initial_size)

    def __str__(self):
        return EMPTYSTRING.join((EMPTYSTRING.join((fws, part)) for fws, part in self))

    def reset(self, startval=None):
        if startval is None:
            startval = []
        self[:] = startval
        self._initial_size = 0

    def is_onlyws(self):
        return self._initial_size == 0 and (not self or str(self).isspace())

    def part_count(self):
        return super().__len__()