import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _reduce_to_set(info, reverse, branches):
    new_branches = []
    items = set()
    case_flags = NOCASE
    for b in branches:
        if isinstance(b, (Character, Property, SetBase)):
            if b.case_flags != case_flags:
                Branch._flush_set_members(info, reverse, items, case_flags, new_branches)
                case_flags = b.case_flags
            items.add(b.with_flags(case_flags=NOCASE))
        else:
            Branch._flush_set_members(info, reverse, items, case_flags, new_branches)
            new_branches.append(b)
    Branch._flush_set_members(info, reverse, items, case_flags, new_branches)
    return new_branches