from typing import List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
from .parse import ParseError, parse
def _attr_resolve_block_rule(state: StateCore) -> None:
    """Find attribute block then move its attributes to the next block."""
    i = 0
    len_tokens = len(state.tokens)
    while i < len_tokens:
        if state.tokens[i].type != 'attrs_block':
            i += 1
            continue
        if i + 1 < len_tokens:
            next_token = state.tokens[i + 1]
            if 'class' in state.tokens[i].attrs and 'class' in next_token.attrs:
                state.tokens[i].attrs['class'] = f'{state.tokens[i].attrs['class']} {next_token.attrs['class']}'
            if next_token.type == 'attrs_block':
                for key, value in state.tokens[i].attrs.items():
                    if key == 'class' or key not in next_token.attrs:
                        next_token.attrs[key] = value
            else:
                next_token.attrs.update(state.tokens[i].attrs)
        state.tokens.pop(i)
        len_tokens -= 1