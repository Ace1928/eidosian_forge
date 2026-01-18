import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _fix_full_casefold(characters):
    expanded = [_regex.fold_case(FULL_CASE_FOLDING, c) for c in _regex.get_expand_on_folding()]
    string = _regex.fold_case(FULL_CASE_FOLDING, ''.join((chr(c) for c in characters))).lower()
    chunks = []
    for e in expanded:
        found = string.find(e)
        while found >= 0:
            chunks.append((found, found + len(e)))
            found = string.find(e, found + 1)
    pos = 0
    literals = []
    for start, end in Sequence._merge_chunks(chunks):
        if pos < start:
            literals.append(Literal(characters[pos:start], case_flags=IGNORECASE))
        literals.append(Literal(characters[start:end], case_flags=FULLIGNORECASE))
        pos = end
    if pos < len(characters):
        literals.append(Literal(characters[pos:], case_flags=IGNORECASE))
    return literals