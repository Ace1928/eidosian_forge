import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _split_common_prefix(info, branches):
    alternatives = []
    for b in branches:
        if isinstance(b, Sequence):
            alternatives.append(b.items)
        else:
            alternatives.append([b])
    max_count = min((len(a) for a in alternatives))
    prefix = alternatives[0]
    pos = 0
    end_pos = max_count
    while pos < end_pos and prefix[pos].can_be_affix() and all((a[pos] == prefix[pos] for a in alternatives)):
        pos += 1
    count = pos
    if info.flags & UNICODE:
        count = pos
        while count > 0 and (not all((Branch._can_split(a, count) for a in alternatives))):
            count -= 1
    if count == 0:
        return ([], branches)
    new_branches = []
    for a in alternatives:
        new_branches.append(make_sequence(a[count:]))
    return (prefix[:count], new_branches)