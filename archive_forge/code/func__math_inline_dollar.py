from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml, isWhiteSpace
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from mdit_py_plugins.utils import is_code_block
def _math_inline_dollar(state: StateInline, silent: bool) -> bool:
    """Inline dollar rule.

        - Initial check:
            - check if first character is a $
            - check if the first character is escaped
            - check if the next character is a space (if not allow_space)
            - check if the next character is a digit (if not allow_digits)
        - Advance one, if allow_double
        - Find closing (advance one, if allow_double)
        - Check closing:
            - check if the previous character is a space (if not allow_space)
            - check if the next character is a digit (if not allow_digits)
        - Check empty content
        """
    if state.src[state.pos] != '$':
        return False
    if not allow_space:
        try:
            if isWhiteSpace(ord(state.src[state.pos + 1])):
                return False
        except IndexError:
            return False
    if not allow_digits:
        try:
            if state.src[state.pos - 1].isdigit():
                return False
        except IndexError:
            pass
    if is_escaped(state, state.pos):
        return False
    try:
        is_double = allow_double and state.src[state.pos + 1] == '$'
    except IndexError:
        return False
    pos = state.pos + 1 + (1 if is_double else 0)
    found_closing = False
    while not found_closing:
        try:
            end = state.src.index('$', pos)
        except ValueError:
            return False
        if is_escaped(state, end):
            pos = end + 1
            continue
        try:
            if is_double and state.src[end + 1] != '$':
                pos = end + 1
                continue
        except IndexError:
            return False
        if is_double:
            end += 1
        found_closing = True
    if not found_closing:
        return False
    if not allow_space:
        try:
            if isWhiteSpace(ord(state.src[end - 1])):
                return False
        except IndexError:
            return False
    if not allow_digits:
        try:
            if state.src[end + 1].isdigit():
                return False
        except IndexError:
            pass
    text = state.src[state.pos + 2:end - 1] if is_double else state.src[state.pos + 1:end]
    if not text:
        return False
    if not silent:
        token = state.push('math_inline_double' if is_double else 'math_inline', 'math', 0)
        token.content = text
        token.markup = '$$' if is_double else '$'
    state.pos = end + 1
    return True