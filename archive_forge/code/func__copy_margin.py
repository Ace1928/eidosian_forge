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
def _copy_margin(self, cli, lazy_screen, new_screen, write_position, move_x, width):
    """
        Copy characters from the margin screen to the real screen.
        """
    xpos = write_position.xpos + move_x
    ypos = write_position.ypos
    margin_write_position = WritePosition(xpos, ypos, width, write_position.height)
    self._copy_body(cli, lazy_screen, new_screen, margin_write_position, 0, width)