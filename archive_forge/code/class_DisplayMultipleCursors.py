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
class DisplayMultipleCursors(Processor):
    """
    When we're in Vi block insert mode, display all the cursors.
    """
    _insert_multiple = ViInsertMultipleMode()

    def __init__(self, buffer_name):
        self.buffer_name = buffer_name

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        buff = cli.buffers[self.buffer_name]
        if self._insert_multiple(cli):
            positions = buff.multiple_cursor_positions
            tokens = explode_tokens(tokens)
            start_pos = document.translate_row_col_to_index(lineno, 0)
            end_pos = start_pos + len(document.lines[lineno])
            token_suffix = (':',) + Token.MultipleCursors.Cursor
            for p in positions:
                if start_pos <= p < end_pos:
                    column = source_to_display(p - start_pos)
                    token, text = tokens[column]
                    token += token_suffix
                    tokens[column] = (token, text)
                elif p == end_pos:
                    tokens.append((token_suffix, ' '))
            return Transformation(tokens)
        else:
            return Transformation(tokens)