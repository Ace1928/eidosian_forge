from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
@staticmethod
def _stripws_default(tlist):
    last_was_ws = False
    is_first_char = True
    for token in tlist.tokens:
        if token.is_whitespace:
            token.value = '' if last_was_ws or is_first_char else ' '
        last_was_ws = token.is_whitespace
        is_first_char = False