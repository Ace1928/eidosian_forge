from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .utils import token_list_to_text
class ScrollbarMargin(Margin):
    """
    Margin displaying a scrollbar.

    :param display_arrows: Display scroll up/down arrows.
    """

    def __init__(self, display_arrows=False):
        self.display_arrows = to_cli_filter(display_arrows)

    def get_width(self, cli, ui_content):
        return 1

    def create_margin(self, cli, window_render_info, width, height):
        total_height = window_render_info.content_height
        display_arrows = self.display_arrows(cli)
        window_height = window_render_info.window_height
        if display_arrows:
            window_height -= 2
        try:
            items_per_row = float(total_height) / min(total_height, window_height)
        except ZeroDivisionError:
            return []
        else:

            def is_scroll_button(row):
                """ True if we should display a button on this row. """
                current_row_middle = int((row + 0.5) * items_per_row)
                return current_row_middle in window_render_info.displayed_lines
            result = []
            if display_arrows:
                result.extend([(Token.Scrollbar.Arrow, '^'), (Token.Scrollbar, '\n')])
            for i in range(window_height):
                if is_scroll_button(i):
                    result.append((Token.Scrollbar.Button, ' '))
                else:
                    result.append((Token.Scrollbar, ' '))
                result.append((Token, '\n'))
            if display_arrows:
                result.append((Token.Scrollbar.Arrow, 'v'))
            return result