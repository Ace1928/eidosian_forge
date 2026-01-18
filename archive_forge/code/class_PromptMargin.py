from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .utils import token_list_to_text
class PromptMargin(Margin):
    """
    Create margin that displays a prompt.
    This can display one prompt at the first line, and a continuation prompt
    (e.g, just dots) on all the following lines.

    :param get_prompt_tokens: Callable that takes a CommandLineInterface as
        input and returns a list of (Token, type) tuples to be shown as the
        prompt at the first line.
    :param get_continuation_tokens: Callable that takes a CommandLineInterface
        and a width as input and returns a list of (Token, type) tuples for the
        next lines of the input.
    :param show_numbers: (bool or :class:`~prompt_toolkit.filters.CLIFilter`)
        Display line numbers instead of the continuation prompt.
    """

    def __init__(self, get_prompt_tokens, get_continuation_tokens=None, show_numbers=False):
        assert callable(get_prompt_tokens)
        assert get_continuation_tokens is None or callable(get_continuation_tokens)
        show_numbers = to_cli_filter(show_numbers)
        self.get_prompt_tokens = get_prompt_tokens
        self.get_continuation_tokens = get_continuation_tokens
        self.show_numbers = show_numbers

    def get_width(self, cli, ui_content):
        """ Width to report to the `Window`. """
        text = token_list_to_text(self.get_prompt_tokens(cli))
        return get_cwidth(text)

    def create_margin(self, cli, window_render_info, width, height):
        tokens = self.get_prompt_tokens(cli)[:]
        if self.get_continuation_tokens:
            tokens2 = list(self.get_continuation_tokens(cli, width))
        else:
            tokens2 = []
        show_numbers = self.show_numbers(cli)
        last_y = None
        for y in window_render_info.displayed_lines[1:]:
            tokens.append((Token, '\n'))
            if show_numbers:
                if y != last_y:
                    tokens.append((Token.LineNumber, ('%i ' % (y + 1)).rjust(width)))
            else:
                tokens.extend(tokens2)
            last_y = y
        return tokens