import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_version_many(tokenizer: Tokenizer) -> str:
    """
    version_many = (SPECIFIER (WS? COMMA WS? SPECIFIER)*)?
    """
    parsed_specifiers = ''
    while tokenizer.check('SPECIFIER'):
        span_start = tokenizer.position
        parsed_specifiers += tokenizer.read().text
        if tokenizer.check('VERSION_PREFIX_TRAIL', peek=True):
            tokenizer.raise_syntax_error('.* suffix can only be used with `==` or `!=` operators', span_start=span_start, span_end=tokenizer.position + 1)
        if tokenizer.check('VERSION_LOCAL_LABEL_TRAIL', peek=True):
            tokenizer.raise_syntax_error('Local version label can only be used with `==` or `!=` operators', span_start=span_start, span_end=tokenizer.position)
        tokenizer.consume('WS')
        if not tokenizer.check('COMMA'):
            break
        parsed_specifiers += tokenizer.read().text
        tokenizer.consume('WS')
    return parsed_specifiers