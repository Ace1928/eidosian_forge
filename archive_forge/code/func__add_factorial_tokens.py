from tokenize import (generate_tokens, untokenize, TokenError,
from keyword import iskeyword
import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Tuple as tTuple, Dict as tDict, Any, Callable, \
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min
def _add_factorial_tokens(name: str, result: List[TOKEN]) -> List[TOKEN]:
    if result == [] or result[-1][1] == '(':
        raise TokenError()
    beginning = [(NAME, name), (OP, '(')]
    end = [(OP, ')')]
    diff = 0
    length = len(result)
    for index, token in enumerate(result[::-1]):
        toknum, tokval = token
        i = length - index - 1
        if tokval == ')':
            diff += 1
        elif tokval == '(':
            diff -= 1
        if diff == 0:
            if i - 1 >= 0 and result[i - 1][0] == NAME:
                return result[:i - 1] + beginning + result[i - 1:] + end
            else:
                return result[:i] + beginning + result[i:] + end
    return result