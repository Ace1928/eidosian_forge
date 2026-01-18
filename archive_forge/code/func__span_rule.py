from typing import List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
from .parse import ParseError, parse
def _span_rule(state: StateInline, silent: bool) -> bool:
    if state.src[state.pos] != '[':
        return False
    maximum = state.posMax
    labelStart = state.pos + 1
    labelEnd = state.md.helpers.parseLinkLabel(state, state.pos, False)
    if labelEnd < 0:
        return False
    pos = labelEnd + 1
    if pos >= maximum:
        return False
    try:
        new_pos, attrs = parse(state.src[pos:])
    except ParseError:
        return False
    pos += new_pos + 1
    if not silent:
        state.pos = labelStart
        state.posMax = labelEnd
        token = state.push('span_open', 'span', 1)
        token.attrs = attrs
        state.md.inline.tokenize(state)
        token = state.push('span_close', 'span', -1)
    state.pos = pos
    state.posMax = maximum
    return True