import re
from ..util import strip_end
def _parse_def_item(block, m):
    head = m.group('def_list_head')
    for line in head.splitlines():
        yield {'type': 'def_list_head', 'text': line}
    src = m.group(0)
    end = len(head)
    m = DD_START_RE.search(src, end)
    start = m.start()
    prev_blank_line = src[end:start] == '\n'
    while m:
        m = DD_START_RE.search(src, start + 1)
        if not m:
            break
        end = m.start()
        text = src[start:end].replace(':', ' ', 1)
        children = _process_text(block, text, prev_blank_line)
        prev_blank_line = bool(HAS_BLANK_LINE_RE.search(text))
        yield {'type': 'def_list_item', 'children': children}
        start = end
    text = src[start:].replace(':', ' ', 1)
    children = _process_text(block, text, prev_blank_line)
    yield {'type': 'def_list_item', 'children': children}