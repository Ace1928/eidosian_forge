import re
from .core import BlockState
from .util import (
def _get_list_bullet(c):
    if c == '.':
        bullet = '\\d{0,9}\\.'
    elif c == ')':
        bullet = '\\d{0,9}\\)'
    elif c == '*':
        bullet = '\\*'
    elif c == '+':
        bullet = '\\+'
    else:
        bullet = '-'
    return bullet