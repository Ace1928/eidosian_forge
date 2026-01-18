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
class BeforeInput(Processor):
    """
    Insert tokens before the input.

    :param get_tokens: Callable that takes a
        :class:`~prompt_toolkit.interface.CommandLineInterface` and returns the
        list of tokens to be inserted.
    """

    def __init__(self, get_tokens):
        assert callable(get_tokens)
        self.get_tokens = get_tokens

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        if lineno == 0:
            tokens_before = self.get_tokens(cli)
            tokens = tokens_before + tokens
            shift_position = token_list_len(tokens_before)
            source_to_display = lambda i: i + shift_position
            display_to_source = lambda i: i - shift_position
        else:
            source_to_display = None
            display_to_source = None
        return Transformation(tokens, source_to_display=source_to_display, display_to_source=display_to_source)

    @classmethod
    def static(cls, text, token=Token):
        """
        Create a :class:`.BeforeInput` instance that always inserts the same
        text.
        """

        def get_static_tokens(cli):
            return [(token, text)]
        return cls(get_static_tokens)

    def __repr__(self):
        return '%s(get_tokens=%r)' % (self.__class__.__name__, self.get_tokens)