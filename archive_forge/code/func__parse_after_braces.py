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
def _parse_after_braces(self, tokens: list, inside_enclosure: bool=False):
    op_dict: dict
    changed: bool = False
    lines: list = []
    self._util_remove_newlines(lines, tokens, inside_enclosure)
    for op_type, grouping_strat, op_dict in reversed(self._mathematica_op_precedence):
        if '*' in op_dict:
            self._util_add_missing_asterisks(tokens)
        size: int = len(tokens)
        pointer: int = 0
        while pointer < size:
            token = tokens[pointer]
            if isinstance(token, str) and token in op_dict:
                op_name: str | Callable = op_dict[token]
                node: list
                first_index: int
                if isinstance(op_name, str):
                    node = [op_name]
                    first_index = 1
                else:
                    node = []
                    first_index = 0
                if token in ('+', '-') and op_type == self.PREFIX and (pointer > 0) and (not self._is_op(tokens[pointer - 1])):
                    pointer += 1
                    continue
                if op_type == self.INFIX:
                    if pointer == 0 or pointer == size - 1 or self._is_op(tokens[pointer - 1]) or self._is_op(tokens[pointer + 1]):
                        pointer += 1
                        continue
                changed = True
                tokens[pointer] = node
                if op_type == self.INFIX:
                    arg1 = tokens.pop(pointer - 1)
                    arg2 = tokens.pop(pointer)
                    if token == '/':
                        arg2 = self._get_inv(arg2)
                    elif token == '-':
                        arg2 = self._get_neg(arg2)
                    pointer -= 1
                    size -= 2
                    node.append(arg1)
                    node_p = node
                    if grouping_strat == self.FLAT:
                        while pointer + 2 < size and self._check_op_compatible(tokens[pointer + 1], token):
                            node_p.append(arg2)
                            other_op = tokens.pop(pointer + 1)
                            arg2 = tokens.pop(pointer + 1)
                            if other_op == '/':
                                arg2 = self._get_inv(arg2)
                            elif other_op == '-':
                                arg2 = self._get_neg(arg2)
                            size -= 2
                        node_p.append(arg2)
                    elif grouping_strat == self.RIGHT:
                        while pointer + 2 < size and tokens[pointer + 1] == token:
                            node_p.append([op_name, arg2])
                            node_p = node_p[-1]
                            tokens.pop(pointer + 1)
                            arg2 = tokens.pop(pointer + 1)
                            size -= 2
                        node_p.append(arg2)
                    elif grouping_strat == self.LEFT:
                        while pointer + 1 < size and tokens[pointer + 1] == token:
                            if isinstance(op_name, str):
                                node_p[first_index] = [op_name, node_p[first_index], arg2]
                            else:
                                node_p[first_index] = op_name(node_p[first_index], arg2)
                            tokens.pop(pointer + 1)
                            arg2 = tokens.pop(pointer + 1)
                            size -= 2
                        node_p.append(arg2)
                    else:
                        node.append(arg2)
                elif op_type == self.PREFIX:
                    assert grouping_strat is None
                    if pointer == size - 1 or self._is_op(tokens[pointer + 1]):
                        tokens[pointer] = self._missing_arguments_default[token]()
                    else:
                        node.append(tokens.pop(pointer + 1))
                        size -= 1
                elif op_type == self.POSTFIX:
                    assert grouping_strat is None
                    if pointer == 0 or self._is_op(tokens[pointer - 1]):
                        tokens[pointer] = self._missing_arguments_default[token]()
                    else:
                        node.append(tokens.pop(pointer - 1))
                        pointer -= 1
                        size -= 1
                if isinstance(op_name, Callable):
                    op_call: Callable = typing.cast(Callable, op_name)
                    new_node = op_call(*node)
                    node.clear()
                    if isinstance(new_node, list):
                        node.extend(new_node)
                    else:
                        tokens[pointer] = new_node
            pointer += 1
    if len(tokens) > 1 or (len(lines) == 0 and len(tokens) == 0):
        if changed:
            return self._parse_after_braces(tokens, inside_enclosure)
        raise SyntaxError('unable to create a single AST for the expression')
    if len(lines) > 0:
        if tokens[0] and tokens[0][0] == 'CompoundExpression':
            tokens = tokens[0][1:]
        compound_expression = ['CompoundExpression', *lines, *tokens]
        return compound_expression
    return tokens[0]