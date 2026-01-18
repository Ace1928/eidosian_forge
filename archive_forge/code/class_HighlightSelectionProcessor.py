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
class HighlightSelectionProcessor(Processor):
    """
    Processor that highlights the selection in the document.
    """

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        selected_token = (':',) + Token.SelectedText
        selection_at_line = document.selection_range_at_line(lineno)
        if selection_at_line:
            from_, to = selection_at_line
            from_ = source_to_display(from_)
            to = source_to_display(to)
            tokens = explode_tokens(tokens)
            if from_ == 0 and to == 0 and (len(tokens) == 0):
                return Transformation([(Token.SelectedText, ' ')])
            else:
                for i in range(from_, to + 1):
                    if i < len(tokens):
                        old_token, old_text = tokens[i]
                        tokens[i] = (old_token + selected_token, old_text)
        return Transformation(tokens)