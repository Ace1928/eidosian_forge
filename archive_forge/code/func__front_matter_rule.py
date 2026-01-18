from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def _front_matter_rule(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    marker_chr = '-'
    min_markers = 3
    auto_closed = False
    start = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]
    src_len = len(state.src)
    if startLine != 0 or state.src[0] != marker_chr:
        return False
    pos = start + 1
    while pos <= maximum and pos < src_len:
        if state.src[pos] != marker_chr:
            break
        pos += 1
    marker_count = pos - start
    if marker_count < min_markers:
        return False
    if silent:
        return True
    nextLine = startLine
    while True:
        nextLine += 1
        if nextLine >= endLine:
            return False
        if state.src[start:maximum] == '...':
            break
        start = state.bMarks[nextLine] + state.tShift[nextLine]
        maximum = state.eMarks[nextLine]
        if start < maximum and state.sCount[nextLine] < state.blkIndent:
            break
        if state.src[start] != marker_chr:
            continue
        if is_code_block(state, nextLine):
            continue
        pos = start + 1
        while pos < maximum:
            if state.src[pos] != marker_chr:
                break
            pos += 1
        if pos - start < marker_count:
            continue
        pos = state.skipSpaces(pos)
        if pos < maximum:
            continue
        auto_closed = True
        break
    old_parent = state.parentType
    old_line_max = state.lineMax
    state.parentType = 'container'
    state.lineMax = nextLine
    token = state.push('front_matter', '', 0)
    token.hidden = True
    token.markup = marker_chr * min_markers
    token.content = state.src[state.bMarks[startLine + 1]:state.eMarks[nextLine - 1]]
    token.block = True
    state.parentType = old_parent
    state.lineMax = old_line_max
    state.line = nextLine + (1 if auto_closed else 0)
    token.map = [startLine, state.line]
    return True