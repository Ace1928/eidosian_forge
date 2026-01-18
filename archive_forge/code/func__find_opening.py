from typing import List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
from .parse import ParseError, parse
def _find_opening(tokens: List[Token], index: int) -> Optional[int]:
    """Find the opening token index, if the token is closing."""
    if tokens[index].nesting != -1:
        return index
    level = 0
    while index >= 0:
        level += tokens[index].nesting
        if level == 0:
            return index
        index -= 1
    return None