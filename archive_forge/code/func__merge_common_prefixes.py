import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _merge_common_prefixes(info, reverse, branches):
    prefixed = defaultdict(list)
    order = {}
    new_branches = []
    for b in branches:
        if Branch._is_simple_character(b):
            prefixed[b.value].append([b])
            order.setdefault(b.value, len(order))
        elif isinstance(b, Sequence) and b.items and Branch._is_simple_character(b.items[0]):
            prefixed[b.items[0].value].append(b.items)
            order.setdefault(b.items[0].value, len(order))
        else:
            Branch._flush_char_prefix(info, reverse, prefixed, order, new_branches)
            new_branches.append(b)
    Branch._flush_char_prefix(info, prefixed, order, new_branches)
    return new_branches