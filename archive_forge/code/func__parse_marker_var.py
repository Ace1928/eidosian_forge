import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_marker_var(tokenizer: Tokenizer) -> MarkerVar:
    """
    marker_var = VARIABLE | QUOTED_STRING
    """
    if tokenizer.check('VARIABLE'):
        return process_env_var(tokenizer.read().text.replace('.', '_'))
    elif tokenizer.check('QUOTED_STRING'):
        return process_python_str(tokenizer.read().text)
    else:
        tokenizer.raise_syntax_error(message='Expected a marker variable or quoted string')