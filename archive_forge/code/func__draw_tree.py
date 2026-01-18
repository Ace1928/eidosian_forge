from __future__ import annotations
import sys
from io import StringIO
from json import JSONEncoder, loads
from typing import TYPE_CHECKING
def _draw_tree(node, prefix: str, child_iter: Callable, text_str: Callable):
    buf = StringIO()
    children = list(child_iter(node))
    if prefix:
        buf.write(prefix[:-3])
        buf.write('  +--')
    buf.write(text_str(node))
    buf.write('\n')
    for index, child in enumerate(children):
        sub_prefix = prefix + '   ' if index + 1 == len(children) else prefix + '  |'
        buf.write(_draw_tree(child, sub_prefix, child_iter, text_str))
    return buf.getvalue()