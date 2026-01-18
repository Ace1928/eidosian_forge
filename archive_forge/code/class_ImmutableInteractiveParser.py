from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
class ImmutableInteractiveParser(InteractiveParser):
    """Same as ``InteractiveParser``, but operations create a new instance instead
    of changing it in-place.
    """
    result = None

    def __hash__(self):
        return hash((self.parser_state, self.lexer_thread))

    def feed_token(self, token):
        c = copy(self)
        c.result = InteractiveParser.feed_token(c, token)
        return c

    def exhaust_lexer(self):
        """Try to feed the rest of the lexer state into the parser.

        Note that this returns a new ImmutableInteractiveParser and does not feed an '$END' Token"""
        cursor = self.as_mutable()
        cursor.exhaust_lexer()
        return cursor.as_immutable()

    def as_mutable(self):
        """Convert to an ``InteractiveParser``."""
        p = copy(self)
        return InteractiveParser(p.parser, p.parser_state, p.lexer_thread)