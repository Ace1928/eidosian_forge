from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable
import sympy
from sympy import Mul, Add, Pow, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning
def _from_mathematica_to_tokens(self, code: str):
    tokenizer = self._get_tokenizer()
    code_splits: list[str | list] = []
    while True:
        string_start = code.find('"')
        if string_start == -1:
            if len(code) > 0:
                code_splits.append(code)
            break
        match_end = re.search('(?<!\\\\)"', code[string_start + 1:])
        if match_end is None:
            raise SyntaxError('mismatch in string "  " expression')
        string_end = string_start + match_end.start() + 1
        if string_start > 0:
            code_splits.append(code[:string_start])
        code_splits.append(['_Str', code[string_start + 1:string_end].replace('\\"', '"')])
        code = code[string_end + 1:]
    for i, code_split in enumerate(code_splits):
        if isinstance(code_split, list):
            continue
        while True:
            pos_comment_start = code_split.find('(*')
            if pos_comment_start == -1:
                break
            pos_comment_end = code_split.find('*)')
            if pos_comment_end == -1 or pos_comment_end < pos_comment_start:
                raise SyntaxError('mismatch in comment (*  *) code')
            code_split = code_split[:pos_comment_start] + code_split[pos_comment_end + 2:]
        code_splits[i] = code_split
    token_lists = [tokenizer.findall(i) if isinstance(i, str) and i.isascii() else [i] for i in code_splits]
    tokens = [j for i in token_lists for j in i]
    while tokens and tokens[0] == '\n':
        tokens.pop(0)
    while tokens and tokens[-1] == '\n':
        tokens.pop(-1)
    return tokens