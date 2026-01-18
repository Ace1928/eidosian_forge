import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class ZeroWidthBase(RegexBase):

    def __init__(self, positive=True):
        RegexBase.__init__(self)
        self.positive = bool(positive)
        self._key = (self.__class__, self.positive)

    def get_firstset(self, reverse):
        return set([None])

    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if fuzzy:
            flags |= FUZZY_OP
        if reverse:
            flags |= REVERSE_OP
        return [(self._opcode, flags)]

    def dump(self, indent, reverse):
        print('{}{} {}'.format(INDENT * indent, self._op_name, POS_TEXT[self.positive]))

    def max_width(self):
        return 0