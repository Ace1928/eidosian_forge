import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_requirement_details(tokenizer: Tokenizer) -> Tuple[str, str, Optional[MarkerList]]:
    """
    requirement_details = AT URL (WS requirement_marker?)?
                        | specifier WS? (requirement_marker)?
    """
    specifier = ''
    url = ''
    marker = None
    if tokenizer.check('AT'):
        tokenizer.read()
        tokenizer.consume('WS')
        url_start = tokenizer.position
        url = tokenizer.expect('URL', expected='URL after @').text
        if tokenizer.check('END', peek=True):
            return (url, specifier, marker)
        tokenizer.expect('WS', expected='whitespace after URL')
        if tokenizer.check('END', peek=True):
            return (url, specifier, marker)
        marker = _parse_requirement_marker(tokenizer, span_start=url_start, after='URL and whitespace')
    else:
        specifier_start = tokenizer.position
        specifier = _parse_specifier(tokenizer)
        tokenizer.consume('WS')
        if tokenizer.check('END', peek=True):
            return (url, specifier, marker)
        marker = _parse_requirement_marker(tokenizer, span_start=specifier_start, after='version specifier' if specifier else 'name and no valid version specifier')
    return (url, specifier, marker)