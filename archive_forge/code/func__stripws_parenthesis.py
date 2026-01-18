from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
def _stripws_parenthesis(self, tlist):
    while tlist.tokens[1].is_whitespace:
        tlist.tokens.pop(1)
    while tlist.tokens[-2].is_whitespace:
        tlist.tokens.pop(-2)
    self._stripws_default(tlist)