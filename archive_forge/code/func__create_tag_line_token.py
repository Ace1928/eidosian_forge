import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Comment, Name, Text, Punctuation, String, Keyword
def _create_tag_line_token(inner_pattern, inner_token, ignore_case=False):
    return (_create_tag_line_pattern(inner_pattern, ignore_case=ignore_case), bygroups(Keyword.Declaration, Text.Whitespace, inner_token, Text.Whitespace, Punctuation, Text.Whitespace, String, Text.Whitespace))