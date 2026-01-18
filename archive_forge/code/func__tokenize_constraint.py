from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def _tokenize_constraint(string, variable_names):
    lparen_re = '\\('
    rparen_re = '\\)'
    op_re = '|'.join([re.escape(op.token_type) for op in _ops])
    num_re = '[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?'
    whitespace_re = '\\s+'
    variable_names = sorted(variable_names, key=len, reverse=True)
    variable_re = '|'.join([re.escape(n) for n in variable_names])
    lexicon = [(lparen_re, _token_maker(Token.LPAREN, string)), (rparen_re, _token_maker(Token.RPAREN, string)), (op_re, _token_maker('__OP__', string)), (variable_re, _token_maker('VARIABLE', string)), (num_re, _token_maker('NUMBER', string)), (whitespace_re, None)]
    scanner = re.Scanner(lexicon)
    tokens, leftover = scanner.scan(string)
    if leftover:
        offset = len(string) - len(leftover)
        raise PatsyError('unrecognized token in constraint', Origin(string, offset, offset + 1))
    return tokens