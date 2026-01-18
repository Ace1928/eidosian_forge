import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_marker_op(tokenizer: Tokenizer) -> Op:
    """
    marker_op = IN | NOT IN | OP
    """
    if tokenizer.check('IN'):
        tokenizer.read()
        return Op('in')
    elif tokenizer.check('NOT'):
        tokenizer.read()
        tokenizer.expect('WS', expected="whitespace after 'not'")
        tokenizer.expect('IN', expected="'in' after 'not'")
        return Op('not in')
    elif tokenizer.check('OP'):
        return Op(tokenizer.read().text)
    else:
        return tokenizer.raise_syntax_error('Expected marker operator, one of <=, <, !=, ==, >=, >, ~=, ===, in, not in')