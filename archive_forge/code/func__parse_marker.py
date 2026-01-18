import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_marker(tokenizer: Tokenizer) -> MarkerList:
    """
    marker = marker_atom (BOOLOP marker_atom)+
    """
    expression = [_parse_marker_atom(tokenizer)]
    while tokenizer.check('BOOLOP'):
        token = tokenizer.read()
        expr_right = _parse_marker_atom(tokenizer)
        expression.extend((token.text, expr_right))
    return expression