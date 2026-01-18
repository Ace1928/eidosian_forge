import re
from ._base import DirectiveParser, BaseDirective
def _process_directive(self, block, marker, start, state):
    mlen = len(marker)
    cursor_start = start + len(marker)
    _end_pattern = '^ {0,3}' + marker[0] + '{' + str(mlen) + ',}[ \\t]*(?:\\n|$)'
    _end_re = re.compile(_end_pattern, re.M)
    _end_m = _end_re.search(state.src, cursor_start)
    if _end_m:
        text = state.src[cursor_start:_end_m.start()]
        end_pos = _end_m.end()
    else:
        text = state.src[cursor_start:]
        end_pos = state.cursor_max
    m = _directive_re.match(text)
    if not m:
        return
    self.parse_method(block, m, state)
    return end_pos