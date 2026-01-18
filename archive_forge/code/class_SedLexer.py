import re
from bisect import bisect
from pygments.lexer import RegexLexer, bygroups, default, include, this, using
from pygments.lexers.python import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, \
class SedLexer(RegexLexer):
    """
    Lexer for Sed script files.
    """
    name = 'Sed'
    aliases = ['sed', 'gsed', 'ssed']
    filenames = ['*.sed', '*.[gs]sed']
    mimetypes = ['text/x-sed']
    flags = re.MULTILINE
    _inside_delims = '((?:(?:\\\\[^\\n]|[^\\\\])*?\\\\\\n)*?(?:\\\\.|[^\\\\])*?)'
    tokens = {'root': [('\\s+', Whitespace), ('#.*$', Comment.Single), ('[0-9]+', Number.Integer), ('\\$', Operator), ('[{};,!]', Punctuation), ('[dDFgGhHlnNpPqQxz=]', Keyword), ('([berRtTvwW:])([^;\\n]*)', bygroups(Keyword, String.Single)), ('([aci])((?:.*?\\\\\\n)*(?:.*?[^\\\\]$))', bygroups(Keyword, String.Double)), ('([qQ])([0-9]*)', bygroups(Keyword, Number.Integer)), ('(/)' + _inside_delims + '(/)', bygroups(Punctuation, String.Regex, Punctuation)), ('(\\\\(.))' + _inside_delims + '(\\2)', bygroups(Punctuation, None, String.Regex, Punctuation)), ('(y)(.)' + _inside_delims + '(\\2)' + _inside_delims + '(\\2)', bygroups(Keyword, Punctuation, String.Single, Punctuation, String.Single, Punctuation)), ('(s)(.)' + _inside_delims + '(\\2)' + _inside_delims + '(\\2)((?:[gpeIiMm]|[0-9])*)', bygroups(Keyword, Punctuation, String.Regex, Punctuation, String.Single, Punctuation, Keyword))]}