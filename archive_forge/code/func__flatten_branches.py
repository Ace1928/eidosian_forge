import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _flatten_branches(info, reverse, branches):
    new_branches = []
    for b in branches:
        b = b.optimise(info, reverse)
        if isinstance(b, Branch):
            new_branches.extend(b.branches)
        else:
            new_branches.append(b)
    return new_branches