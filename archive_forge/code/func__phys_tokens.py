from __future__ import annotations
import ast
import io
import keyword
import re
import sys
import token
import tokenize
from typing import Iterable
from coverage import env
from coverage.types import TLineNo, TSourceTokenLines
def _phys_tokens(toks: TokenInfos) -> TokenInfos:
    """Return all physical tokens, even line continuations.

    tokenize.generate_tokens() doesn't return a token for the backslash that
    continues lines.  This wrapper provides those tokens so that we can
    re-create a faithful representation of the original source.

    Returns the same values as generate_tokens()

    """
    last_line: str | None = None
    last_lineno = -1
    last_ttext: str = ''
    for ttype, ttext, (slineno, scol), (elineno, ecol), ltext in toks:
        if last_lineno != elineno:
            if last_line and last_line.endswith('\\\n'):
                inject_backslash = True
                if last_ttext.endswith('\\'):
                    inject_backslash = False
                elif ttype == token.STRING:
                    if '\n' in ttext and ttext.split('\n', 1)[0][-1] == '\\':
                        inject_backslash = False
                if inject_backslash:
                    ccol = len(last_line.split('\n')[-2]) - 1
                    yield tokenize.TokenInfo(99999, '\\\n', (slineno, ccol), (slineno, ccol + 2), last_line)
            last_line = ltext
        if ttype not in (tokenize.NEWLINE, tokenize.NL):
            last_ttext = ttext
        yield tokenize.TokenInfo(ttype, ttext, (slineno, scol), (elineno, ecol), ltext)
        last_lineno = elineno