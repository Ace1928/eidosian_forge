from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from mdit_py_plugins.utils import is_code_block
def _substitution_inline(state: StateInline, silent: bool) -> bool:
    try:
        if state.src[state.pos] != start_delimiter or state.src[state.pos + 1] != start_delimiter:
            return False
    except IndexError:
        return False
    pos = state.pos + 2
    found_closing = False
    while True:
        try:
            end = state.src.index(end_delimiter, pos)
        except ValueError:
            return False
        try:
            if state.src[end + 1] == end_delimiter:
                found_closing = True
                break
        except IndexError:
            return False
        pos = end + 2
    if not found_closing:
        return False
    text = state.src[state.pos + 2:end].strip()
    state.pos = end + 2
    if silent:
        return True
    token = state.push('substitution_inline', 'span', 0)
    token.block = False
    token.content = text
    token.attrSet('class', 'substitution')
    token.attrSet('text', text)
    token.markup = f'{start_delimiter}{end_delimiter}'
    return True