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
class AfterInput(Processor):
    """
    Insert tokens after the input.

    :param get_tokens: Callable that takes a
        :class:`~prompt_toolkit.interface.CommandLineInterface` and returns the
        list of tokens to be appended.
    """

    def __init__(self, get_tokens):
        assert callable(get_tokens)
        self.get_tokens = get_tokens

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        if lineno == document.line_count - 1:
            return Transformation(tokens=tokens + self.get_tokens(cli))
        else:
            return Transformation(tokens=tokens)

    @classmethod
    def static(cls, text, token=Token):
        """
        Create a :class:`.AfterInput` instance that always inserts the same
        text.
        """

        def get_static_tokens(cli):
            return [(token, text)]
        return cls(get_static_tokens)

    def __repr__(self):
        return '%s(get_tokens=%r)' % (self.__class__.__name__, self.get_tokens)