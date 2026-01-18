from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
class Parenthesis(TokenList):
    """Tokens between parenthesis."""
    M_OPEN = (T.Punctuation, '(')
    M_CLOSE = (T.Punctuation, ')')

    @property
    def _groupable_tokens(self):
        return self.tokens[1:-1]