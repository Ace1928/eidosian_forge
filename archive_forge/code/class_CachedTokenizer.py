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
class CachedTokenizer:
    """A one-element cache around tokenize.generate_tokens.

    When reporting, coverage.py tokenizes files twice, once to find the
    structure of the file, and once to syntax-color it.  Tokenizing is
    expensive, and easily cached.

    This is a one-element cache so that our twice-in-a-row tokenizing doesn't
    actually tokenize twice.

    """

    def __init__(self) -> None:
        self.last_text: str | None = None
        self.last_tokens: list[tokenize.TokenInfo] = []

    def generate_tokens(self, text: str) -> TokenInfos:
        """A stand-in for `tokenize.generate_tokens`."""
        if text != self.last_text:
            self.last_text = text
            readline = io.StringIO(text).readline
            try:
                self.last_tokens = list(tokenize.generate_tokens(readline))
            except:
                self.last_text = None
                raise
        return self.last_tokens