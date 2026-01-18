from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def _token_matching(self, funcs, start=0, end=None, reverse=False):
    """next token that match functions"""
    if start is None:
        return None
    if not isinstance(funcs, (list, tuple)):
        funcs = (funcs,)
    if reverse:
        assert end is None
        for idx in range(start - 2, -1, -1):
            token = self.tokens[idx]
            for func in funcs:
                if func(token):
                    return (idx, token)
    else:
        for idx, token in enumerate(self.tokens[start:end], start=start):
            for func in funcs:
                if func(token):
                    return (idx, token)
    return (None, None)