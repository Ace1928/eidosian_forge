from __future__ import annotations
import itertools
import re
from enum import Enum
from typing import Hashable, TypeVar
from prompt_toolkit.cache import SimpleCache
from .base import (
from .named_colors import NAMED_COLORS
def _parse_style_str(style_str: str) -> Attrs:
    """
    Take a style string, e.g.  'bg:red #88ff00 class:title'
    and return a `Attrs` instance.
    """
    if 'noinherit' in style_str:
        attrs = DEFAULT_ATTRS
    else:
        attrs = _EMPTY_ATTRS
    for part in style_str.split():
        if part == 'noinherit':
            pass
        elif part == 'bold':
            attrs = attrs._replace(bold=True)
        elif part == 'nobold':
            attrs = attrs._replace(bold=False)
        elif part == 'italic':
            attrs = attrs._replace(italic=True)
        elif part == 'noitalic':
            attrs = attrs._replace(italic=False)
        elif part == 'underline':
            attrs = attrs._replace(underline=True)
        elif part == 'nounderline':
            attrs = attrs._replace(underline=False)
        elif part == 'strike':
            attrs = attrs._replace(strike=True)
        elif part == 'nostrike':
            attrs = attrs._replace(strike=False)
        elif part == 'blink':
            attrs = attrs._replace(blink=True)
        elif part == 'noblink':
            attrs = attrs._replace(blink=False)
        elif part == 'reverse':
            attrs = attrs._replace(reverse=True)
        elif part == 'noreverse':
            attrs = attrs._replace(reverse=False)
        elif part == 'hidden':
            attrs = attrs._replace(hidden=True)
        elif part == 'nohidden':
            attrs = attrs._replace(hidden=False)
        elif part in ('roman', 'sans', 'mono'):
            pass
        elif part.startswith('border:'):
            pass
        elif part.startswith('[') and part.endswith(']'):
            pass
        elif part.startswith('bg:'):
            attrs = attrs._replace(bgcolor=parse_color(part[3:]))
        elif part.startswith('fg:'):
            attrs = attrs._replace(color=parse_color(part[3:]))
        else:
            attrs = attrs._replace(color=parse_color(part))
    return attrs