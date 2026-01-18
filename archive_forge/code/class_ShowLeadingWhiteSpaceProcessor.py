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
class ShowLeadingWhiteSpaceProcessor(Processor):
    """
    Make leading whitespace visible.

    :param get_char: Callable that takes a :class:`CommandLineInterface`
        instance and returns one character.
    :param token: Token to be used.
    """

    def __init__(self, get_char=None, token=Token.LeadingWhiteSpace):
        assert get_char is None or callable(get_char)
        if get_char is None:

            def get_char(cli):
                if '·'.encode(cli.output.encoding(), 'replace') == b'?':
                    return '.'
                else:
                    return '·'
        self.token = token
        self.get_char = get_char

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        if tokens and token_list_to_text(tokens).startswith(' '):
            t = (self.token, self.get_char(cli))
            tokens = explode_tokens(tokens)
            for i in range(len(tokens)):
                if tokens[i][1] == ' ':
                    tokens[i] = t
                else:
                    break
        return Transformation(tokens)