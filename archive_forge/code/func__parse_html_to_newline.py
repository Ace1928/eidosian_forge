import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def _parse_html_to_newline(state, newline):
    m = newline.search(state.src, state.cursor)
    if m:
        end_pos = m.start()
        text = state.get_text(end_pos)
    else:
        text = state.src[state.cursor:]
        end_pos = state.cursor_max
    state.append_token({'type': 'block_html', 'raw': text})
    return end_pos