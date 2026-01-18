import re
from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
class StripTrailingSemicolonFilter:

    def process(self, stmt):
        while stmt.tokens and (stmt.tokens[-1].is_whitespace or stmt.tokens[-1].value == ';'):
            stmt.tokens.pop()
        return stmt