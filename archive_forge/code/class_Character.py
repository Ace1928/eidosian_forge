import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Character(RegexBase):
    _opcode = {(NOCASE, False): OP.CHARACTER, (IGNORECASE, False): OP.CHARACTER_IGN, (FULLCASE, False): OP.CHARACTER, (FULLIGNORECASE, False): OP.CHARACTER_IGN, (NOCASE, True): OP.CHARACTER_REV, (IGNORECASE, True): OP.CHARACTER_IGN_REV, (FULLCASE, True): OP.CHARACTER_REV, (FULLIGNORECASE, True): OP.CHARACTER_IGN_REV}

    def __init__(self, value, positive=True, case_flags=NOCASE, zerowidth=False):
        RegexBase.__init__(self)
        self.value = value
        self.positive = bool(positive)
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self.zerowidth = bool(zerowidth)
        if self.positive and self.case_flags & FULLIGNORECASE == FULLIGNORECASE:
            self.folded = _regex.fold_case(FULL_CASE_FOLDING, chr(self.value))
        else:
            self.folded = chr(self.value)
        self._key = (self.__class__, self.value, self.positive, self.case_flags, self.zerowidth)

    def rebuild(self, positive, case_flags, zerowidth):
        return Character(self.value, positive, case_flags, zerowidth)

    def optimise(self, info, reverse, in_set=False):
        return self

    def get_firstset(self, reverse):
        return set([self])

    def has_simple_start(self):
        return True

    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if self.zerowidth:
            flags |= ZEROWIDTH_OP
        if fuzzy:
            flags |= FUZZY_OP
        code = PrecompiledCode([self._opcode[self.case_flags, reverse], flags, self.value])
        if len(self.folded) > 1:
            code = Branch([code, String([ord(c) for c in self.folded], case_flags=self.case_flags)])
        return code.compile(reverse, fuzzy)

    def dump(self, indent, reverse):
        display = ascii(chr(self.value)).lstrip('bu')
        print('{}CHARACTER {} {}{}'.format(INDENT * indent, POS_TEXT[self.positive], display, CASE_TEXT[self.case_flags]))

    def matches(self, ch):
        return (ch == self.value) == self.positive

    def max_width(self):
        return len(self.folded)

    def get_required_string(self, reverse):
        if not self.positive:
            return (1, None)
        self.folded_characters = tuple((ord(c) for c in self.folded))
        return (0, self)