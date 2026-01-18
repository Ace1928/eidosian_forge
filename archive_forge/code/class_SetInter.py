import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class SetInter(SetBase):
    _opcode = {(NOCASE, False): OP.SET_INTER, (IGNORECASE, False): OP.SET_INTER_IGN, (FULLCASE, False): OP.SET_INTER, (FULLIGNORECASE, False): OP.SET_INTER_IGN, (NOCASE, True): OP.SET_INTER_REV, (IGNORECASE, True): OP.SET_INTER_IGN_REV, (FULLCASE, True): OP.SET_INTER_REV, (FULLIGNORECASE, True): OP.SET_INTER_IGN_REV}
    _op_name = 'SET_INTER'

    def optimise(self, info, reverse, in_set=False):
        items = []
        for m in self.items:
            m = m.optimise(info, reverse, in_set=True)
            if isinstance(m, SetInter) and m.positive:
                items.extend(m.items)
            else:
                items.append(m)
        if len(items) == 1:
            return items[0].with_flags(case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
        self.items = tuple(items)
        return self._handle_case_folding(info, in_set)

    def matches(self, ch):
        m = all((i.matches(ch) for i in self.items))
        return m == self.positive