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
def _copy_body(self, cli, ui_content, new_screen, write_position, move_x, width, vertical_scroll=0, horizontal_scroll=0, has_focus=False, wrap_lines=False, highlight_lines=False, vertical_scroll_2=0, always_hide_cursor=False):
    """
        Copy the UIContent into the output screen.
        """
    xpos = write_position.xpos + move_x
    ypos = write_position.ypos
    line_count = ui_content.line_count
    new_buffer = new_screen.data_buffer
    empty_char = _CHAR_CACHE['', Token]
    ZeroWidthEscape = Token.ZeroWidthEscape
    visible_line_to_row_col = {}
    rowcol_to_yx = {}
    default_char = ui_content.default_char
    if default_char:
        for y in range(ypos, ypos + write_position.height):
            new_buffer_row = new_buffer[y]
            for x in range(xpos, xpos + width):
                new_buffer_row[x] = default_char

    def copy():
        y = -vertical_scroll_2
        lineno = vertical_scroll
        while y < write_position.height and lineno < line_count:
            line = ui_content.get_line(lineno)
            col = 0
            x = -horizontal_scroll
            visible_line_to_row_col[y] = (lineno, horizontal_scroll)
            new_buffer_row = new_buffer[y + ypos]
            for token, text in line:
                if token == ZeroWidthEscape:
                    new_screen.zero_width_escapes[y + ypos][x + xpos] += text
                    continue
                for c in text:
                    char = _CHAR_CACHE[c, token]
                    char_width = char.width
                    if wrap_lines and x + char_width > width:
                        visible_line_to_row_col[y + 1] = (lineno, visible_line_to_row_col[y][1] + x)
                        y += 1
                        x = -horizontal_scroll
                        new_buffer_row = new_buffer[y + ypos]
                        if y >= write_position.height:
                            return y
                    if x >= 0 and y >= 0 and (x < write_position.width):
                        new_buffer_row[x + xpos] = char
                        if char_width > 1:
                            for i in range(1, char_width):
                                new_buffer_row[x + xpos + i] = empty_char
                        elif char_width == 0 and x - 1 >= 0:
                            prev_char = new_buffer_row[x + xpos - 1]
                            char2 = _CHAR_CACHE[prev_char.char + c, prev_char.token]
                            new_buffer_row[x + xpos - 1] = char2
                        rowcol_to_yx[lineno, col] = (y + ypos, x + xpos)
                    col += 1
                    x += char_width
            lineno += 1
            y += 1
        return y
    y = copy()

    def cursor_pos_to_screen_pos(row, col):
        """ Translate row/col from UIContent to real Screen coordinates. """
        try:
            y, x = rowcol_to_yx[row, col]
        except KeyError:
            return Point(y=0, x=0)
        else:
            return Point(y=y, x=x)
    if ui_content.cursor_position:
        screen_cursor_position = cursor_pos_to_screen_pos(ui_content.cursor_position.y, ui_content.cursor_position.x)
        if has_focus:
            new_screen.cursor_position = screen_cursor_position
            if always_hide_cursor:
                new_screen.show_cursor = False
            else:
                new_screen.show_cursor = ui_content.show_cursor
            self._highlight_digraph(cli, new_screen)
        if highlight_lines:
            self._highlight_cursorlines(cli, new_screen, screen_cursor_position, xpos, ypos, width, write_position.height)
    if has_focus and ui_content.cursor_position:
        self._show_input_processor_key_buffer(cli, new_screen)
    if not new_screen.menu_position and ui_content.menu_position:
        new_screen.menu_position = cursor_pos_to_screen_pos(ui_content.menu_position.y, ui_content.menu_position.x)
    new_screen.height = max(new_screen.height, ypos + write_position.height)
    return (visible_line_to_row_col, rowcol_to_yx)