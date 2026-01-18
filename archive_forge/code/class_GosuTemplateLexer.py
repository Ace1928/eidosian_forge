import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class GosuTemplateLexer(Lexer):
    """
    For Gosu templates.

    .. versionadded:: 1.5
    """
    name = 'Gosu Template'
    aliases = ['gst']
    filenames = ['*.gst']
    mimetypes = ['text/x-gosu-template']

    def get_tokens_unprocessed(self, text):
        lexer = GosuLexer()
        stack = ['templateText']
        for item in lexer.get_tokens_unprocessed(text, stack):
            yield item