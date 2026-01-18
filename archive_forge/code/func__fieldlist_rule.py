from contextlib import contextmanager
from typing import Iterator, Optional, Tuple
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def _fieldlist_rule(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    if is_code_block(state, startLine):
        return False
    posAfterName, name_text = parseNameMarker(state, startLine)
    if posAfterName < 0:
        return False
    if silent:
        return True
    token = state.push('field_list_open', 'dl', 1)
    token.attrSet('class', 'field-list')
    token.map = listLines = [startLine, 0]
    nextLine = startLine
    with set_parent_type(state, 'fieldlist'):
        while nextLine < endLine:
            token = state.push('fieldlist_name_open', 'dt', 1)
            token.map = [startLine, startLine]
            token = state.push('inline', '', 0)
            token.map = [startLine, startLine]
            token.content = name_text
            token.children = []
            token = state.push('fieldlist_name_close', 'dt', -1)
            pos = posAfterName
            maximum: int = state.eMarks[nextLine]
            first_line_body_indent = state.sCount[nextLine] + posAfterName - (state.bMarks[startLine] + state.tShift[startLine])
            while pos < maximum:
                ch = state.src[pos]
                if ch == '\t':
                    first_line_body_indent += 4 - (first_line_body_indent + state.bsCount[nextLine]) % 4
                elif ch == ' ':
                    first_line_body_indent += 1
                else:
                    break
                pos += 1
            contentStart = pos
            block_indent: Optional[int] = None
            _line = startLine + 1
            while _line < endLine:
                if state.bMarks[_line] + state.tShift[_line] < state.eMarks[_line]:
                    if state.tShift[_line] <= 0:
                        break
                    block_indent = state.tShift[_line] if block_indent is None else min(block_indent, state.tShift[_line])
                _line += 1
            has_first_line = contentStart < maximum
            if block_indent is None:
                if not has_first_line:
                    block_indent = 2
                else:
                    block_indent = first_line_body_indent
            else:
                block_indent = min(block_indent, first_line_body_indent)
            token = state.push('fieldlist_body_open', 'dd', 1)
            token.map = [startLine, startLine]
            with temp_state_changes(state, startLine):
                diff = 0
                if has_first_line and block_indent < first_line_body_indent:
                    diff = first_line_body_indent - block_indent
                    state.src = state.src[:contentStart - diff] + ' ' * diff + state.src[contentStart:]
                state.tShift[startLine] = contentStart - diff - state.bMarks[startLine]
                state.sCount[startLine] = first_line_body_indent - diff
                state.blkIndent = block_indent
                state.md.block.tokenize(state, startLine, endLine)
            state.push('fieldlist_body_close', 'dd', -1)
            nextLine = startLine = state.line
            token.map[1] = nextLine
            if nextLine >= endLine:
                break
            contentStart = state.bMarks[startLine]
            if state.sCount[nextLine] < state.blkIndent:
                break
            if is_code_block(state, startLine):
                break
            posAfterName, name_text = parseNameMarker(state, startLine)
            if posAfterName < 0:
                break
        token = state.push('field_list_close', 'dl', -1)
        listLines[1] = nextLine
        state.line = nextLine
    return True