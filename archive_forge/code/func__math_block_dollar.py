from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml, isWhiteSpace
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from mdit_py_plugins.utils import is_code_block
def _math_block_dollar(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    if is_code_block(state, startLine):
        return False
    haveEndMarker = False
    startPos = state.bMarks[startLine] + state.tShift[startLine]
    end = state.eMarks[startLine]
    if startPos + 2 > end:
        return False
    if state.src[startPos] != '$' or state.src[startPos + 1] != '$':
        return False
    nextLine = startLine
    label = None
    lineText = state.src[startPos:end]
    if len(lineText.strip()) > 3:
        if lineText.strip().endswith('$$'):
            haveEndMarker = True
            end = end - 2 - (len(lineText) - len(lineText.strip()))
        elif allow_labels:
            eqnoMatch = DOLLAR_EQNO_REV.match(lineText[::-1])
            if eqnoMatch:
                haveEndMarker = True
                label = eqnoMatch.group(1)[::-1]
                end = end - eqnoMatch.end()
    if not haveEndMarker:
        while True:
            nextLine += 1
            if nextLine >= endLine:
                break
            start = state.bMarks[nextLine] + state.tShift[nextLine]
            end = state.eMarks[nextLine]
            lineText = state.src[start:end]
            if lineText.strip().endswith('$$'):
                haveEndMarker = True
                end = end - 2 - (len(lineText) - len(lineText.strip()))
                break
            if lineText.strip() == '' and (not allow_blank_lines):
                break
            if allow_labels:
                eqnoMatch = DOLLAR_EQNO_REV.match(lineText[::-1])
                if eqnoMatch:
                    haveEndMarker = True
                    label = eqnoMatch.group(1)[::-1]
                    end = end - eqnoMatch.end()
                    break
    if not haveEndMarker:
        return False
    state.line = nextLine + (1 if haveEndMarker else 0)
    token = state.push('math_block_label' if label else 'math_block', 'math', 0)
    token.block = True
    token.content = state.src[startPos + 2:end]
    token.markup = '$$'
    token.map = [startLine, state.line]
    if label:
        token.info = label if label_normalizer is None else label_normalizer(label)
    return True