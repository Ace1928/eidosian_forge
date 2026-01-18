import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_requirement_marker(tokenizer: Tokenizer, *, span_start: int, after: str) -> MarkerList:
    """
    requirement_marker = SEMICOLON marker WS?
    """
    if not tokenizer.check('SEMICOLON'):
        tokenizer.raise_syntax_error(f'Expected end or semicolon (after {after})', span_start=span_start)
    tokenizer.read()
    marker = _parse_marker(tokenizer)
    tokenizer.consume('WS')
    return marker