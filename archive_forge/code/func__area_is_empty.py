from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from .controls import UIControl, TokenListControl, UIContent
from .dimension import LayoutDimension, sum_layout_dimensions, max_layout_dimensions
from .margins import Margin
from .screen import Point, WritePosition, _CHAR_CACHE
from .utils import token_list_to_text, explode_tokens
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import to_cli_filter, ViInsertMode, EmacsInsertMode
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from prompt_toolkit.utils import take_using_weights, get_cwidth
def _area_is_empty(self, screen, write_position):
    """
        Return True when the area below the write position is still empty.
        (For floats that should not hide content underneath.)
        """
    wp = write_position
    Transparent = Token.Transparent
    for y in range(wp.ypos, wp.ypos + wp.height):
        if y in screen.data_buffer:
            row = screen.data_buffer[y]
            for x in range(wp.xpos, wp.xpos + wp.width):
                c = row[x]
                if c.char != ' ' or c.token != Transparent:
                    return False
    return True