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
def _mouse_handler(self, cli, mouse_event):
    """
        Mouse handler. Called when the UI control doesn't handle this
        particular event.
        """
    if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
        self._scroll_down(cli)
    elif mouse_event.event_type == MouseEventType.SCROLL_UP:
        self._scroll_up(cli)