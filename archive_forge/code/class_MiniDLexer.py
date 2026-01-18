from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Keyword, Name, String, \
class MiniDLexer(CrocLexer):
    """
    For MiniD source. MiniD is now known as Croc.
    """
    name = 'MiniD'
    filenames = []
    aliases = ['minid']
    mimetypes = ['text/x-minidsrc']