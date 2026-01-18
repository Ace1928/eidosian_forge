import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class SetSymDiff(SetBase):
    _opcode = {(NOCASE, False): OP.SET_SYM_DIFF, (IGNORECASE, False): OP.SET_SYM_DIFF_IGN, (FULLCASE, False): OP.SET_SYM_DIFF, (FULLIGNORECASE, False): OP.SET_SYM_DIFF_IGN, (NOCASE, True): OP.SET_SYM_DIFF_REV, (IGNORECASE, True): OP.SET_SYM_DIFF_IGN_REV, (FULLCASE, True): OP.SET_SYM_DIFF_REV, (FULLIGNORECASE, True): OP.SET_SYM_DIFF_IGN_REV}
    _op_name = 'SET_SYM_DIFF'

    def optimise(self, info, reverse, in_set=False):
        items = []
        for m in self.items:
            m = m.optimise(info, reverse, in_set=True)
            if isinstance(m, SetSymDiff) and m.positive:
                items.extend(m.items)
            else:
                items.append(m)
        if len(items) == 1:
            return items[0].with_flags(case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
        self.items = tuple(items)
        return self._handle_case_folding(info, in_set)

    def matches(self, ch):
        m = False
        for i in self.items:
            m = m != i.matches(ch)
        return m == self.positive