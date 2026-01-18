from __future__ import annotations
import ast
import io
import os
import sys
import tokenize
from collections.abc import (
from os.path import relpath
from textwrap import dedent
from tokenize import COMMENT, NAME, OP, STRING, generate_tokens
from typing import TYPE_CHECKING, Any
from babel.util import parse_encoding, parse_future_flags, pathmatch
def _strip_comment_tags(comments: MutableSequence[str], tags: Iterable[str]):
    """Helper function for `extract` that strips comment tags from strings
    in a list of comment lines.  This functions operates in-place.
    """

    def _strip(line: str):
        for tag in tags:
            if line.startswith(tag):
                return line[len(tag):].strip()
        return line
    comments[:] = map(_strip, comments)