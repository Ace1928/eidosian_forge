import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_specifier(tokenizer: Tokenizer) -> str:
    """
    specifier = LEFT_PARENTHESIS WS? version_many WS? RIGHT_PARENTHESIS
              | WS? version_many WS?
    """
    with tokenizer.enclosing_tokens('LEFT_PARENTHESIS', 'RIGHT_PARENTHESIS', around='version specifier'):
        tokenizer.consume('WS')
        parsed_specifiers = _parse_version_many(tokenizer)
        tokenizer.consume('WS')
    return parsed_specifiers