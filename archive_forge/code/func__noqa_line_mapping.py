from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
@functools.cached_property
def _noqa_line_mapping(self) -> dict[int, str]:
    """Map from line number to the line we'll search for `noqa` in."""
    try:
        file_tokens = self.file_tokens
    except (tokenize.TokenError, SyntaxError):
        return {}
    else:
        ret = {}
        min_line = len(self.lines) + 2
        max_line = -1
        for tp, _, (s_line, _), (e_line, _), _ in file_tokens:
            if tp == tokenize.ENDMARKER or tp == tokenize.DEDENT:
                continue
            min_line = min(min_line, s_line)
            max_line = max(max_line, e_line)
            if tp in (tokenize.NL, tokenize.NEWLINE):
                ret.update(self._noqa_line_range(min_line, max_line))
                min_line = len(self.lines) + 2
                max_line = -1
        return ret