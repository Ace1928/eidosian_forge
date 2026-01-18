import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _flush_set_members(info, reverse, items, case_flags, new_branches):
    if not items:
        return
    if len(items) == 1:
        item = list(items)[0]
    else:
        item = SetUnion(info, list(items)).optimise(info, reverse)
    new_branches.append(item.with_flags(case_flags=case_flags))
    items.clear()