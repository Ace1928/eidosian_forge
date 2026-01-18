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
def _from_tokens_to_fullformlist(self, tokens: list):
    stack: list[list] = [[]]
    open_seq = []
    pointer: int = 0
    while pointer < len(tokens):
        token = tokens[pointer]
        if token in self._enclosure_open:
            stack[-1].append(token)
            open_seq.append(token)
            stack.append([])
        elif token == ',':
            if len(stack[-1]) == 0 and stack[-2][-1] == open_seq[-1]:
                raise SyntaxError('%s cannot be followed by comma ,' % open_seq[-1])
            stack[-1] = self._parse_after_braces(stack[-1])
            stack.append([])
        elif token in self._enclosure_close:
            ind = self._enclosure_close.index(token)
            if self._enclosure_open[ind] != open_seq[-1]:
                unmatched_enclosure = SyntaxError('unmatched enclosure')
                if token == ']]' and open_seq[-1] == '[':
                    if open_seq[-2] == '[':
                        tokens.insert(pointer + 1, ']')
                    elif open_seq[-2] == '[[':
                        if tokens[pointer + 1] == ']':
                            tokens[pointer + 1] = ']]'
                        elif tokens[pointer + 1] == ']]':
                            tokens[pointer + 1] = ']]'
                            tokens.insert(pointer + 2, ']')
                        else:
                            raise unmatched_enclosure
                else:
                    raise unmatched_enclosure
            if len(stack[-1]) == 0 and stack[-2][-1] == '(':
                raise SyntaxError('( ) not valid syntax')
            last_stack = self._parse_after_braces(stack[-1], True)
            stack[-1] = last_stack
            new_stack_element = []
            while stack[-1][-1] != open_seq[-1]:
                new_stack_element.append(stack.pop())
            new_stack_element.reverse()
            if open_seq[-1] == '(' and len(new_stack_element) != 1:
                raise SyntaxError('( must be followed by one expression, %i detected' % len(new_stack_element))
            stack[-1].append(new_stack_element)
            open_seq.pop(-1)
        else:
            stack[-1].append(token)
        pointer += 1
    assert len(stack) == 1
    return self._parse_after_braces(stack[0])