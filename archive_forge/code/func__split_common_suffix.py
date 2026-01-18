import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _split_common_suffix(info, branches):
    alternatives = []
    for b in branches:
        if isinstance(b, Sequence):
            alternatives.append(b.items)
        else:
            alternatives.append([b])
    max_count = min((len(a) for a in alternatives))
    suffix = alternatives[0]
    pos = -1
    end_pos = -1 - max_count
    while pos > end_pos and suffix[pos].can_be_affix() and all((a[pos] == suffix[pos] for a in alternatives)):
        pos -= 1
    count = -1 - pos
    if info.flags & UNICODE:
        while count > 0 and (not all((Branch._can_split_rev(a, count) for a in alternatives))):
            count -= 1
    if count == 0:
        return ([], branches)
    new_branches = []
    for a in alternatives:
        new_branches.append(make_sequence(a[:-count]))
    return (suffix[-count:], new_branches)