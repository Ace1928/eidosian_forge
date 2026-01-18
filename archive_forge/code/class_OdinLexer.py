from pygments.lexer import RegexLexer, include, bygroups, using, default
from pygments.token import Text, Comment, Name, Literal, Number, String, \
class OdinLexer(AtomsLexer):
    """
    Lexer for ODIN syntax.

    .. versionadded:: 2.1
    """
    name = 'ODIN'
    aliases = ['odin']
    filenames = ['*.odin']
    mimetypes = ['text/odin']
    tokens = {'path': [('>', Punctuation, '#pop'), ('[a-z_]\\w*', Name.Class), ('/', Punctuation), ('\\[', Punctuation, 'key'), ('\\s*,\\s*', Punctuation, '#pop'), ('\\s+', Text, '#pop')], 'key': [include('values'), ('\\]', Punctuation, '#pop')], 'type_cast': [('\\)', Punctuation, '#pop'), ('[^)]+', Name.Class)], 'root': [include('whitespace'), ('([Tt]rue|[Ff]alse)', Literal), include('values'), ('/', Punctuation, 'path'), ('\\[', Punctuation, 'key'), ('[a-z_]\\w*', Name.Class), ('=', Operator), ('\\(', Punctuation, 'type_cast'), (',', Punctuation), ('<', Punctuation), ('>', Punctuation), (';', Punctuation)]}