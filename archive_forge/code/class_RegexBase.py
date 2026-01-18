import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class RegexBase:

    def __init__(self):
        self._key = self.__class__

    def with_flags(self, positive=None, case_flags=None, zerowidth=None):
        if positive is None:
            positive = self.positive
        else:
            positive = bool(positive)
        if case_flags is None:
            case_flags = self.case_flags
        else:
            case_flags = CASE_FLAGS_COMBINATIONS[case_flags & CASE_FLAGS]
        if zerowidth is None:
            zerowidth = self.zerowidth
        else:
            zerowidth = bool(zerowidth)
        if positive == self.positive and case_flags == self.case_flags and (zerowidth == self.zerowidth):
            return self
        return self.rebuild(positive, case_flags, zerowidth)

    def fix_groups(self, pattern, reverse, fuzzy):
        pass

    def optimise(self, info, reverse):
        return self

    def pack_characters(self, info):
        return self

    def remove_captures(self):
        return self

    def is_atomic(self):
        return True

    def can_be_affix(self):
        return True

    def contains_group(self):
        return False

    def get_firstset(self, reverse):
        raise _FirstSetError()

    def has_simple_start(self):
        return False

    def compile(self, reverse=False, fuzzy=False):
        return self._compile(reverse, fuzzy)

    def is_empty(self):
        return False

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return type(self) is type(other) and self._key == other._key

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_required_string(self, reverse):
        return (self.max_width(), None)