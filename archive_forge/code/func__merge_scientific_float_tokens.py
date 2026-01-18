from typing import (
import cmath
import re
import numpy as np
import sympy
def _merge_scientific_float_tokens(tokens: Iterable[str]) -> List[str]:
    tokens = list(tokens)
    i = 0
    while 'e' in tokens[i + 1:]:
        i = tokens.index('e', i + 1)
        s = i - 1
        e = i + 1
        if not re.match('[0-9]', str(tokens[s])):
            continue
        if re.match('[+-]', str(tokens[e])):
            e += 1
        if re.match('[0-9]', str(tokens[e])):
            e += 1
            tokens[s:e] = [''.join(tokens[s:e])]
            i -= 1
    return tokens