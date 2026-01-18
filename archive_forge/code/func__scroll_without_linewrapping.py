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
def _scroll_without_linewrapping(self, ui_content, width, height, cli):
    """
        Scroll to make sure the cursor position is visible and that we maintain
        the requested scroll offset.

        Set `self.horizontal_scroll/vertical_scroll`.
        """
    cursor_position = ui_content.cursor_position or Point(0, 0)
    self.vertical_scroll_2 = 0
    if ui_content.line_count == 0:
        self.vertical_scroll = 0
        self.horizontal_scroll = 0
        return
    else:
        current_line_text = token_list_to_text(ui_content.get_line(cursor_position.y))

    def do_scroll(current_scroll, scroll_offset_start, scroll_offset_end, cursor_pos, window_size, content_size):
        """ Scrolling algorithm. Used for both horizontal and vertical scrolling. """
        scroll_offset_start = int(min(scroll_offset_start, window_size / 2, cursor_pos))
        scroll_offset_end = int(min(scroll_offset_end, window_size / 2, content_size - 1 - cursor_pos))
        if current_scroll < 0:
            current_scroll = 0
        if not self.allow_scroll_beyond_bottom(cli) and current_scroll > content_size - window_size:
            current_scroll = max(0, content_size - window_size)
        if current_scroll > cursor_pos - scroll_offset_start:
            current_scroll = max(0, cursor_pos - scroll_offset_start)
        if current_scroll < cursor_pos + 1 - window_size + scroll_offset_end:
            current_scroll = cursor_pos + 1 - window_size + scroll_offset_end
        return current_scroll
    if self.get_vertical_scroll:
        self.vertical_scroll = self.get_vertical_scroll(self)
        assert isinstance(self.vertical_scroll, int)
    if self.get_horizontal_scroll:
        self.horizontal_scroll = self.get_horizontal_scroll(self)
        assert isinstance(self.horizontal_scroll, int)
    offsets = self.scroll_offsets
    self.vertical_scroll = do_scroll(current_scroll=self.vertical_scroll, scroll_offset_start=offsets.top, scroll_offset_end=offsets.bottom, cursor_pos=ui_content.cursor_position.y, window_size=height, content_size=ui_content.line_count)
    self.horizontal_scroll = do_scroll(current_scroll=self.horizontal_scroll, scroll_offset_start=offsets.left, scroll_offset_end=offsets.right, cursor_pos=get_cwidth(current_line_text[:ui_content.cursor_position.x]), window_size=width, content_size=max(get_cwidth(current_line_text), self.horizontal_scroll + width))