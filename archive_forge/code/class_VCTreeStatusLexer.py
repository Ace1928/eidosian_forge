from pygments.lexer import RegexLexer, include, bygroups
from pygments.token import Generic, Comment, String, Text, Keyword, Name, \
class VCTreeStatusLexer(RegexLexer):
    """
    For colorizing output of version control status commands, like "hg
    status" or "svn status".

    .. versionadded:: 2.0
    """
    name = 'VCTreeStatus'
    aliases = ['vctreestatus']
    filenames = []
    mimetypes = []
    tokens = {'root': [('^A  \\+  C\\s+', Generic.Error), ('^A\\s+\\+?\\s+', String), ('^M\\s+', Generic.Inserted), ('^C\\s+', Generic.Error), ('^D\\s+', Generic.Deleted), ('^[?!]\\s+', Comment.Preproc), ('      >\\s+.*\\n', Comment.Preproc), ('.*\\n', Text)]}