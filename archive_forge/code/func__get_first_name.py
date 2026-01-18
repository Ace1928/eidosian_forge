from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def _get_first_name(self, idx=None, reverse=False, keywords=False, real_name=False):
    """Returns the name of the first token with a name"""
    tokens = self.tokens[idx:] if idx else self.tokens
    tokens = reversed(tokens) if reverse else tokens
    types = [T.Name, T.Wildcard, T.String.Symbol]
    if keywords:
        types.append(T.Keyword)
    for token in tokens:
        if token.ttype in types:
            return remove_quotes(token.value)
        elif isinstance(token, (Identifier, Function)):
            return token.get_real_name() if real_name else token.get_name()