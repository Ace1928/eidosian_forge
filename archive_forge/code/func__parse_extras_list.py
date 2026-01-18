import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_extras_list(tokenizer: Tokenizer) -> List[str]:
    """
    extras_list = identifier (wsp* ',' wsp* identifier)*
    """
    extras: List[str] = []
    if not tokenizer.check('IDENTIFIER'):
        return extras
    extras.append(tokenizer.read().text)
    while True:
        tokenizer.consume('WS')
        if tokenizer.check('IDENTIFIER', peek=True):
            tokenizer.raise_syntax_error('Expected comma between extra names')
        elif not tokenizer.check('COMMA'):
            break
        tokenizer.read()
        tokenizer.consume('WS')
        extra_token = tokenizer.expect('IDENTIFIER', expected='extra name after comma')
        extras.append(extra_token.text)
    return extras