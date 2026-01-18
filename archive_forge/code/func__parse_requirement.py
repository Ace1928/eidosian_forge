import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_requirement(tokenizer: Tokenizer) -> ParsedRequirement:
    """
    requirement = WS? IDENTIFIER WS? extras WS? requirement_details
    """
    tokenizer.consume('WS')
    name_token = tokenizer.expect('IDENTIFIER', expected='package name at the start of dependency specifier')
    name = name_token.text
    tokenizer.consume('WS')
    extras = _parse_extras(tokenizer)
    tokenizer.consume('WS')
    url, specifier, marker = _parse_requirement_details(tokenizer)
    tokenizer.expect('END', expected='end of dependency specifier')
    return ParsedRequirement(name, url, extras, specifier, marker)