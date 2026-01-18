import math
import re
from typing import (
import unicodedata
from .parser import Parser
def _dump_dict(obj, skipkeys, ensure_ascii, check_circular, allow_nan, indent, separators, default, sort_keys, quote_keys, trailing_commas, allow_duplicate_keys, seen, level, item_sep, kv_sep, indent_str, end_str):
    if not obj:
        return '{}'
    if sort_keys:
        keys = sorted(obj.keys())
    else:
        keys = obj.keys()
    s = '{' + indent_str
    num_items_added = 0
    new_keys = set()
    for key in keys:
        valid_key, key_str = _dumps(key, skipkeys, ensure_ascii, check_circular, allow_nan, indent, separators, default, sort_keys, quote_keys, trailing_commas, allow_duplicate_keys, seen, level, is_key=True)
        if skipkeys and (not valid_key):
            continue
        if not allow_duplicate_keys:
            if key_str in new_keys:
                raise ValueError(f'duplicate key {repr(key)}')
            new_keys.add(key_str)
        if num_items_added:
            s += item_sep
        s += key_str + kv_sep + _dumps(obj[key], skipkeys, ensure_ascii, check_circular, allow_nan, indent, separators, default, sort_keys, quote_keys, trailing_commas, allow_duplicate_keys, seen, level, is_key=False)[1]
        num_items_added += 1
    s += end_str + '}'
    return s