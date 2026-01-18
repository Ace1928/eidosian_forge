from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from six import with_metaclass
from six.moves import range
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.search_state import SearchState
from prompt_toolkit.selection import SelectionType
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .lexers import Lexer, SimpleLexer
from .processors import Processor
from .screen import Char, Point
from .utils import token_list_width, split_lines, token_list_to_text
import six
import time
class FillControl(UIControl):
    """
    Fill whole control with characters with this token.
    (Also helpful for debugging.)

    :param char: :class:`.Char` instance to use for filling.
    :param get_char: A callable that takes a CommandLineInterface and returns a
        :class:`.Char` object.
    """

    def __init__(self, character=None, token=Token, char=None, get_char=None):
        assert char is None or isinstance(char, Char)
        assert get_char is None or callable(get_char)
        assert not (char and get_char)
        self.char = char
        if character:
            self.character = character
            self.token = token
            self.get_char = lambda cli: Char(character, token)
        elif get_char:
            self.get_char = get_char
        else:
            self.char = self.char or Char()
            self.get_char = lambda cli: self.char
            self.char = char

    def __repr__(self):
        if self.char:
            return '%s(char=%r)' % (self.__class__.__name__, self.char)
        else:
            return '%s(get_char=%r)' % (self.__class__.__name__, self.get_char)

    def reset(self):
        pass

    def has_focus(self, cli):
        return False

    def create_content(self, cli, width, height):

        def get_line(i):
            return []
        return UIContent(get_line=get_line, line_count=100 ** 100, default_char=self.get_char(cli))