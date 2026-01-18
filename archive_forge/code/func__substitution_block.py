from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from mdit_py_plugins.utils import is_code_block
def _substitution_block(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    if is_code_block(state, startLine):
        return False
    startPos = state.bMarks[startLine] + state.tShift[startLine]
    end = state.eMarks[startLine]
    lineText = state.src[startPos:end].strip()
    try:
        if lineText[0] != start_delimiter or lineText[1] != start_delimiter or lineText[-1] != end_delimiter or (lineText[-2] != end_delimiter) or (len(lineText) < 5):
            return False
    except IndexError:
        return False
    text = lineText[2:-2].strip()
    if end_delimiter * 2 in text:
        return False
    state.line = startLine + 1
    if silent:
        return True
    token = state.push('substitution_block', 'div', 0)
    token.block = True
    token.content = text
    token.attrSet('class', 'substitution')
    token.attrSet('text', text)
    token.markup = f'{start_delimiter}{end_delimiter}'
    token.map = [startLine, state.line]
    return True