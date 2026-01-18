import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class GreedyRepeat(RegexBase):
    _opcode = OP.GREEDY_REPEAT
    _op_name = 'GREEDY_REPEAT'

    def __init__(self, subpattern, min_count, max_count):
        RegexBase.__init__(self)
        self.subpattern = subpattern
        self.min_count = min_count
        self.max_count = max_count

    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, reverse, fuzzy)

    def optimise(self, info, reverse):
        subpattern = self.subpattern.optimise(info, reverse)
        return type(self)(subpattern, self.min_count, self.max_count)

    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self

    def remove_captures(self):
        self.subpattern = self.subpattern.remove_captures()
        return self

    def is_atomic(self):
        return self.min_count == self.max_count and self.subpattern.is_atomic()

    def can_be_affix(self):
        return False

    def contains_group(self):
        return self.subpattern.contains_group()

    def get_firstset(self, reverse):
        fs = self.subpattern.get_firstset(reverse)
        if self.min_count == 0:
            fs.add(None)
        return fs

    def _compile(self, reverse, fuzzy):
        repeat = [self._opcode, self.min_count]
        if self.max_count is None:
            repeat.append(UNLIMITED)
        else:
            repeat.append(self.max_count)
        subpattern = self.subpattern.compile(reverse, fuzzy)
        if not subpattern:
            return []
        return [tuple(repeat)] + subpattern + [(OP.END,)]

    def dump(self, indent, reverse):
        if self.max_count is None:
            limit = 'INF'
        else:
            limit = self.max_count
        print('{}{} {} {}'.format(INDENT * indent, self._op_name, self.min_count, limit))
        self.subpattern.dump(indent + 1, reverse)

    def is_empty(self):
        return self.subpattern.is_empty()

    def __eq__(self, other):
        return type(self) is type(other) and (self.subpattern, self.min_count, self.max_count) == (other.subpattern, other.min_count, other.max_count)

    def max_width(self):
        if self.max_count is None:
            return UNLIMITED
        return self.subpattern.max_width() * self.max_count

    def get_required_string(self, reverse):
        max_count = UNLIMITED if self.max_count is None else self.max_count
        if self.min_count == 0:
            w = self.subpattern.max_width() * max_count
            return (min(w, UNLIMITED), None)
        ofs, req = self.subpattern.get_required_string(reverse)
        if req:
            return (ofs, req)
        w = self.subpattern.max_width() * max_count
        return (min(w, UNLIMITED), None)