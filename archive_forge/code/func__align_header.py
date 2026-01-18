from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _align_header(header, alignment, width, visible_width, is_multiline=False, width_fn=None):
    """Pad string header to width chars given known visible_width of the header."""
    if is_multiline:
        header_lines = re.split(_multiline_codes, header)
        padded_lines = [_align_header(h, alignment, width, width_fn(h)) for h in header_lines]
        return '\n'.join(padded_lines)
    ninvisible = len(header) - visible_width
    width += ninvisible
    if alignment == 'left':
        return _padright(width, header)
    elif alignment == 'center':
        return _padboth(width, header)
    elif not alignment:
        return f'{header}'
    else:
        return _padleft(width, header)