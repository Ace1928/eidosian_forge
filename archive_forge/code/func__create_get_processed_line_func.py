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
def _create_get_processed_line_func(self, cli, document):
    """
        Create a function that takes a line number of the current document and
        returns a _ProcessedLine(processed_tokens, source_to_display, display_to_source)
        tuple.
        """

    def transform(lineno, tokens):
        """ Transform the tokens for a given line number. """
        source_to_display_functions = []
        display_to_source_functions = []
        if document.cursor_position_row == lineno:
            cursor_column = document.cursor_position_col
        else:
            cursor_column = None

        def source_to_display(i):
            """ Translate x position from the buffer to the x position in the
                processed token list. """
            for f in source_to_display_functions:
                i = f(i)
            return i
        for p in self.input_processors:
            transformation = p.apply_transformation(cli, document, lineno, source_to_display, tokens)
            tokens = transformation.tokens
            if cursor_column:
                cursor_column = transformation.source_to_display(cursor_column)
            display_to_source_functions.append(transformation.display_to_source)
            source_to_display_functions.append(transformation.source_to_display)

        def display_to_source(i):
            for f in reversed(display_to_source_functions):
                i = f(i)
            return i
        return _ProcessedLine(tokens, source_to_display, display_to_source)

    def create_func():
        get_line = self._get_tokens_for_line_func(cli, document)
        cache = {}

        def get_processed_line(i):
            try:
                return cache[i]
            except KeyError:
                processed_line = transform(i, get_line(i))
                cache[i] = processed_line
                return processed_line
        return get_processed_line
    return create_func()