from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml, unescapeAll
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def _skipCharsStr(state: StateBlock, pos: int, ch: str) -> int:
    """Skip character string from given position."""
    while True:
        try:
            current = state.src[pos]
        except IndexError:
            break
        if current != ch:
            break
        pos += 1
    return pos