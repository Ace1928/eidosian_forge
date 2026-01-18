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
class WindowRenderInfo(object):
    """
    Render information, for the last render time of this control.
    It stores mapping information between the input buffers (in case of a
    :class:`~prompt_toolkit.layout.controls.BufferControl`) and the actual
    render position on the output screen.

    (Could be used for implementation of the Vi 'H' and 'L' key bindings as
    well as implementing mouse support.)

    :param ui_content: The original :class:`.UIContent` instance that contains
        the whole input, without clipping. (ui_content)
    :param horizontal_scroll: The horizontal scroll of the :class:`.Window` instance.
    :param vertical_scroll: The vertical scroll of the :class:`.Window` instance.
    :param window_width: The width of the window that displays the content,
        without the margins.
    :param window_height: The height of the window that displays the content.
    :param configured_scroll_offsets: The scroll offsets as configured for the
        :class:`Window` instance.
    :param visible_line_to_row_col: Mapping that maps the row numbers on the
        displayed screen (starting from zero for the first visible line) to
        (row, col) tuples pointing to the row and column of the :class:`.UIContent`.
    :param rowcol_to_yx: Mapping that maps (row, column) tuples representing
        coordinates of the :class:`UIContent` to (y, x) absolute coordinates at
        the rendered screen.
    """

    def __init__(self, ui_content, horizontal_scroll, vertical_scroll, window_width, window_height, configured_scroll_offsets, visible_line_to_row_col, rowcol_to_yx, x_offset, y_offset, wrap_lines):
        assert isinstance(ui_content, UIContent)
        assert isinstance(horizontal_scroll, int)
        assert isinstance(vertical_scroll, int)
        assert isinstance(window_width, int)
        assert isinstance(window_height, int)
        assert isinstance(configured_scroll_offsets, ScrollOffsets)
        assert isinstance(visible_line_to_row_col, dict)
        assert isinstance(rowcol_to_yx, dict)
        assert isinstance(x_offset, int)
        assert isinstance(y_offset, int)
        assert isinstance(wrap_lines, bool)
        self.ui_content = ui_content
        self.vertical_scroll = vertical_scroll
        self.window_width = window_width
        self.window_height = window_height
        self.configured_scroll_offsets = configured_scroll_offsets
        self.visible_line_to_row_col = visible_line_to_row_col
        self.wrap_lines = wrap_lines
        self._rowcol_to_yx = rowcol_to_yx
        self._x_offset = x_offset
        self._y_offset = y_offset

    @property
    def visible_line_to_input_line(self):
        return dict(((visible_line, rowcol[0]) for visible_line, rowcol in self.visible_line_to_row_col.items()))

    @property
    def cursor_position(self):
        """
        Return the cursor position coordinates, relative to the left/top corner
        of the rendered screen.
        """
        cpos = self.ui_content.cursor_position
        y, x = self._rowcol_to_yx[cpos.y, cpos.x]
        return Point(x=x - self._x_offset, y=y - self._y_offset)

    @property
    def applied_scroll_offsets(self):
        """
        Return a :class:`.ScrollOffsets` instance that indicates the actual
        offset. This can be less than or equal to what's configured. E.g, when
        the cursor is completely at the top, the top offset will be zero rather
        than what's configured.
        """
        if self.displayed_lines[0] == 0:
            top = 0
        else:
            y = self.input_line_to_visible_line[self.ui_content.cursor_position.y]
            top = min(y, self.configured_scroll_offsets.top)
        return ScrollOffsets(top=top, bottom=min(self.ui_content.line_count - self.displayed_lines[-1] - 1, self.configured_scroll_offsets.bottom), left=0, right=0)

    @property
    def displayed_lines(self):
        """
        List of all the visible rows. (Line numbers of the input buffer.)
        The last line may not be entirely visible.
        """
        return sorted((row for row, col in self.visible_line_to_row_col.values()))

    @property
    def input_line_to_visible_line(self):
        """
        Return the dictionary mapping the line numbers of the input buffer to
        the lines of the screen. When a line spans several rows at the screen,
        the first row appears in the dictionary.
        """
        result = {}
        for k, v in self.visible_line_to_input_line.items():
            if v in result:
                result[v] = min(result[v], k)
            else:
                result[v] = k
        return result

    def first_visible_line(self, after_scroll_offset=False):
        """
        Return the line number (0 based) of the input document that corresponds
        with the first visible line.
        """
        if after_scroll_offset:
            return self.displayed_lines[self.applied_scroll_offsets.top]
        else:
            return self.displayed_lines[0]

    def last_visible_line(self, before_scroll_offset=False):
        """
        Like `first_visible_line`, but for the last visible line.
        """
        if before_scroll_offset:
            return self.displayed_lines[-1 - self.applied_scroll_offsets.bottom]
        else:
            return self.displayed_lines[-1]

    def center_visible_line(self, before_scroll_offset=False, after_scroll_offset=False):
        """
        Like `first_visible_line`, but for the center visible line.
        """
        return self.first_visible_line(after_scroll_offset) + (self.last_visible_line(before_scroll_offset) - self.first_visible_line(after_scroll_offset)) // 2

    @property
    def content_height(self):
        """
        The full height of the user control.
        """
        return self.ui_content.line_count

    @property
    def full_height_visible(self):
        """
        True when the full height is visible (There is no vertical scroll.)
        """
        return self.vertical_scroll == 0 and self.last_visible_line() == self.content_height

    @property
    def top_visible(self):
        """
        True when the top of the buffer is visible.
        """
        return self.vertical_scroll == 0

    @property
    def bottom_visible(self):
        """
        True when the bottom of the buffer is visible.
        """
        return self.last_visible_line() == self.content_height - 1

    @property
    def vertical_scroll_percentage(self):
        """
        Vertical scroll as a percentage. (0 means: the top is visible,
        100 means: the bottom is visible.)
        """
        if self.bottom_visible:
            return 100
        else:
            return 100 * self.vertical_scroll // self.content_height

    def get_height_for_line(self, lineno):
        """
        Return the height of the given line.
        (The height that it would take, if this line became visible.)
        """
        if self.wrap_lines:
            return self.ui_content.get_height_for_line(lineno, self.window_width)
        else:
            return 1