from __future__ import unicode_literals
from six.moves import zip_longest, range
from prompt_toolkit.filters import HasCompletions, IsDone, Condition, to_cli_filter
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .containers import Window, HSplit, ConditionalContainer, ScrollOffsets
from .controls import UIControl, UIContent
from .dimension import LayoutDimension
from .margins import ScrollbarMargin
from .screen import Point, Char
import math
def _trim_text(text, max_width):
    """
    Trim the text to `max_width`, append dots when the text is too long.
    Returns (text, width) tuple.
    """
    width = get_cwidth(text)
    if width > max_width:
        if len(text) == width:
            trimmed_text = (text[:max(1, max_width - 3)] + '...')[:max_width]
            return (trimmed_text, len(trimmed_text))
        else:
            trimmed_text = ''
            for c in text:
                if get_cwidth(trimmed_text + c) <= max_width - 3:
                    trimmed_text += c
            trimmed_text += '...'
            return (trimmed_text, get_cwidth(trimmed_text))
    else:
        return (text, width)