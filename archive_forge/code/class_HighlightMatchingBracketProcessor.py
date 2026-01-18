from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.document import Document
from prompt_toolkit.enums import SEARCH_BUFFER
from prompt_toolkit.filters import to_cli_filter, ViInsertMultipleMode
from prompt_toolkit.layout.utils import token_list_to_text
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from .utils import token_list_len, explode_tokens
import re
class HighlightMatchingBracketProcessor(Processor):
    """
    When the cursor is on or right after a bracket, it highlights the matching
    bracket.

    :param max_cursor_distance: Only highlight matching brackets when the
        cursor is within this distance. (From inside a `Processor`, we can't
        know which lines will be visible on the screen. But we also don't want
        to scan the whole document for matching brackets on each key press, so
        we limit to this value.)
    """
    _closing_braces = '])}>'

    def __init__(self, chars='[](){}<>', max_cursor_distance=1000):
        self.chars = chars
        self.max_cursor_distance = max_cursor_distance
        self._positions_cache = SimpleCache(maxsize=8)

    def _get_positions_to_highlight(self, document):
        """
        Return a list of (row, col) tuples that need to be highlighted.
        """
        if document.current_char and document.current_char in self.chars:
            pos = document.find_matching_bracket_position(start_pos=document.cursor_position - self.max_cursor_distance, end_pos=document.cursor_position + self.max_cursor_distance)
        elif document.char_before_cursor and document.char_before_cursor in self._closing_braces and (document.char_before_cursor in self.chars):
            document = Document(document.text, document.cursor_position - 1)
            pos = document.find_matching_bracket_position(start_pos=document.cursor_position - self.max_cursor_distance, end_pos=document.cursor_position + self.max_cursor_distance)
        else:
            pos = None
        if pos:
            pos += document.cursor_position
            row, col = document.translate_index_to_position(pos)
            return [(row, col), (document.cursor_position_row, document.cursor_position_col)]
        else:
            return []

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        key = (cli.render_counter, document.text, document.cursor_position)
        positions = self._positions_cache.get(key, lambda: self._get_positions_to_highlight(document))
        if positions:
            for row, col in positions:
                if row == lineno:
                    col = source_to_display(col)
                    tokens = explode_tokens(tokens)
                    token, text = tokens[col]
                    if col == document.cursor_position_col:
                        token += (':',) + Token.MatchingBracket.Cursor
                    else:
                        token += (':',) + Token.MatchingBracket.Other
                    tokens[col] = (token, text)
        return Transformation(tokens)