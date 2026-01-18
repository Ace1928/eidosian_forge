import re
from .core import BlockState
from .util import (
def _clean_list_item_text(src, continue_width):
    rv = []
    trim_space = ' ' * continue_width
    lines = src.split('\n')
    for line in lines:
        if line.startswith(trim_space):
            line = line.replace(trim_space, '', 1)
            line = expand_tab(line)
            rv.append(line)
        else:
            rv.append(line)
    return '\n'.join(rv)