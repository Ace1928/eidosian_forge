import re
from pygments.lexer import RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class Inform6TemplateLexer(Inform7Lexer):
    """
    For `Inform 6 template
    <http://inform7.com/sources/src/i6template/Woven/index.html>`_ code.

    .. versionadded:: 2.0
    """
    name = 'Inform 6 template'
    aliases = ['i6t']
    filenames = ['*.i6t']

    def get_tokens_unprocessed(self, text, stack=('+i6t-root',)):
        return Inform7Lexer.get_tokens_unprocessed(self, text, stack)