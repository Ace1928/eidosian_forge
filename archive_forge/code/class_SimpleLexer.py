from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
class SimpleLexer(Lexer):
    """
    Lexer that doesn't do any tokenizing and returns the whole input as one token.

    :param token: The `Token` for this lexer.
    """

    def __init__(self, token=Token, default_token=None):
        self.token = token
        if default_token is not None:
            self.token = default_token

    def lex_document(self, cli, document):
        lines = document.lines

        def get_line(lineno):
            """ Return the tokens for the given line. """
            try:
                return [(self.token, lines[lineno])]
            except IndexError:
                return []
        return get_line