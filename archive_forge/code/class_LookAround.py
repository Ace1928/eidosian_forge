import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class LookAround(RegexBase):
    _dir_text = {False: 'AHEAD', True: 'BEHIND'}

    def __init__(self, behind, positive, subpattern):
        RegexBase.__init__(self)
        self.behind = bool(behind)
        self.positive = bool(positive)
        self.subpattern = subpattern

    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, self.behind, fuzzy)

    def optimise(self, info, reverse):
        subpattern = self.subpattern.optimise(info, self.behind)
        if self.positive and subpattern.is_empty():
            return subpattern
        return LookAround(self.behind, self.positive, subpattern)

    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self

    def remove_captures(self):
        return self.subpattern.remove_captures()

    def is_atomic(self):
        return self.subpattern.is_atomic()

    def can_be_affix(self):
        return self.subpattern.can_be_affix()

    def contains_group(self):
        return self.subpattern.contains_group()

    def get_firstset(self, reverse):
        if self.positive and self.behind == reverse:
            return self.subpattern.get_firstset(reverse)
        return set([None])

    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if fuzzy:
            flags |= FUZZY_OP
        if reverse:
            flags |= REVERSE_OP
        return [(OP.LOOKAROUND, flags, int(not self.behind))] + self.subpattern.compile(self.behind) + [(OP.END,)]

    def dump(self, indent, reverse):
        print('{}LOOK{} {}'.format(INDENT * indent, self._dir_text[self.behind], POS_TEXT[self.positive]))
        self.subpattern.dump(indent + 1, self.behind)

    def is_empty(self):
        return self.positive and self.subpattern.is_empty()

    def __eq__(self, other):
        return type(self) is type(other) and (self.behind, self.positive, self.subpattern) == (other.behind, other.positive, other.subpattern)

    def max_width(self):
        return 0