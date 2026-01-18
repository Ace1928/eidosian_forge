import argparse
import collections
import contextlib
import io
import re
import tokenize
from typing import TextIO, Tuple
import untokenize  # type: ignore
import docformatter.encode as _encode
import docformatter.strings as _strings
import docformatter.syntax as _syntax
import docformatter.util as _util
def _do_remove_blank_lines_after_definitions(modified_tokens):
    """Remove blank lines between definitions and docstrings.

    Blank lines between class, method, function, and variable
    definitions and the docstring will be removed.

    Parameters
    ----------
    modified_tokens: list
        The list of tokens created from the docstring.

    Returns
    -------
    modified_tokens: list
        The list of tokens with any blank lines following a variable
        definition removed.
    """
    for _idx, _token in enumerate(modified_tokens):
        if _token[0] == 3:
            j = 1
            while modified_tokens[_idx - j][4] == '\n' and (not modified_tokens[_idx - j - 1][4].strip().endswith('"""')) and (not modified_tokens[_idx - j - 1][4].startswith('#!/')) and ('import' not in modified_tokens[_idx - j - 1][4]):
                modified_tokens.pop(_idx - j)
                j += 1
            j = 2
            while modified_tokens[_idx - j][4] == '\n' and modified_tokens[_idx - j - 2][4].strip().startswith(('def', 'class')):
                modified_tokens.pop(_idx - j)
                j += 1
    return modified_tokens