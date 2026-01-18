import math
import re
from typing import (
import unicodedata
from .parser import Parser
def _walk_ast(el, dictify: Callable[[Iterable[Tuple[str, Any]]], Any], parse_float, parse_int, parse_constant):
    if el == 'None':
        return None
    if el == 'True':
        return True
    if el == 'False':
        return False
    ty, v = el
    if ty == 'number':
        if v.startswith('0x') or v.startswith('0X'):
            return parse_int(v, base=16)
        if '.' in v or 'e' in v or 'E' in v:
            return parse_float(v)
        if 'Infinity' in v or 'NaN' in v:
            return parse_constant(v)
        return parse_int(v)
    if ty == 'string':
        return v
    if ty == 'object':
        pairs = []
        for key, val_expr in v:
            val = _walk_ast(val_expr, dictify, parse_float, parse_int, parse_constant)
            pairs.append((key, val))
        return dictify(pairs)
    if ty == 'array':
        return [_walk_ast(el, dictify, parse_float, parse_int, parse_constant) for el in v]
    raise ValueError('unknown el: ' + el)