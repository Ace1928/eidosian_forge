import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class SetDiff(SetBase):
    _opcode = {(NOCASE, False): OP.SET_DIFF, (IGNORECASE, False): OP.SET_DIFF_IGN, (FULLCASE, False): OP.SET_DIFF, (FULLIGNORECASE, False): OP.SET_DIFF_IGN, (NOCASE, True): OP.SET_DIFF_REV, (IGNORECASE, True): OP.SET_DIFF_IGN_REV, (FULLCASE, True): OP.SET_DIFF_REV, (FULLIGNORECASE, True): OP.SET_DIFF_IGN_REV}
    _op_name = 'SET_DIFF'

    def optimise(self, info, reverse, in_set=False):
        items = self.items
        if len(items) > 2:
            items = [items[0], SetUnion(info, items[1:])]
        if len(items) == 1:
            return items[0].with_flags(case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
        self.items = tuple((m.optimise(info, reverse, in_set=True) for m in items))
        return self._handle_case_folding(info, in_set)

    def matches(self, ch):
        m = self.items[0].matches(ch) and (not self.items[1].matches(ch))
        return m == self.positive