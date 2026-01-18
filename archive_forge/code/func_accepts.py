from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
def accepts(self):
    """Returns the set of possible tokens that will advance the parser into a new valid state."""
    accepts = set()
    conf_no_callbacks = copy(self.parser_state.parse_conf)
    conf_no_callbacks.callbacks = {}
    for t in self.choices():
        if t.isupper():
            new_cursor = copy(self)
            new_cursor.parser_state.parse_conf = conf_no_callbacks
            try:
                new_cursor.feed_token(self.lexer_thread._Token(t, ''))
            except UnexpectedToken:
                pass
            else:
                accepts.add(t)
    return accepts