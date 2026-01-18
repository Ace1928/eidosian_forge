import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
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