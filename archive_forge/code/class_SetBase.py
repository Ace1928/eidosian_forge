import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class SetBase(RegexBase):

    def __init__(self, info, items, positive=True, case_flags=NOCASE, zerowidth=False):
        RegexBase.__init__(self)
        self.info = info
        self.items = tuple(items)
        self.positive = bool(positive)
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self.zerowidth = bool(zerowidth)
        self.char_width = 1
        self._key = (self.__class__, self.items, self.positive, self.case_flags, self.zerowidth)

    def rebuild(self, positive, case_flags, zerowidth):
        return type(self)(self.info, self.items, positive, case_flags, zerowidth).optimise(self.info, False)

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
        code = [(self._opcode[self.case_flags, reverse], flags)]
        for m in self.items:
            code.extend(m.compile())
        code.append((OP.END,))
        return code

    def dump(self, indent, reverse):
        print('{}{} {}{}'.format(INDENT * indent, self._op_name, POS_TEXT[self.positive], CASE_TEXT[self.case_flags]))
        for i in self.items:
            i.dump(indent + 1, reverse)

    def _handle_case_folding(self, info, in_set):
        if not self.positive or not self.case_flags & IGNORECASE or in_set:
            return self
        if not self.info.flags & UNICODE or self.case_flags & FULLIGNORECASE != FULLIGNORECASE:
            return self
        expanding_chars = _regex.get_expand_on_folding()
        items = []
        seen = set()
        for ch in expanding_chars:
            if self.matches(ord(ch)):
                folded = _regex.fold_case(FULL_CASE_FOLDING, ch)
                if folded not in seen:
                    items.append(String([ord(c) for c in folded], case_flags=self.case_flags))
                    seen.add(folded)
        if not items:
            return self
        return Branch([self] + items)

    def max_width(self):
        if not self.positive or not self.case_flags & IGNORECASE:
            return 1
        if not self.info.flags & UNICODE or self.case_flags & FULLIGNORECASE != FULLIGNORECASE:
            return 1
        expanding_chars = _regex.get_expand_on_folding()
        seen = set()
        for ch in expanding_chars:
            if self.matches(ord(ch)):
                folded = _regex.fold_case(FULL_CASE_FOLDING, ch)
                seen.add(folded)
        if not seen:
            return 1
        return max((len(folded) for folded in seen))

    def __del__(self):
        self.info = None