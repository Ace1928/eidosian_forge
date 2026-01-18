from pygments.lexer import RegexLexer, words
from pygments.token import Comment, Keyword, Name, String, Text
class Macaulay2Lexer(RegexLexer):
    """Lexer for Macaulay2, a software system for research in algebraic geometry."""
    name = 'Macaulay2'
    url = 'https://faculty.math.illinois.edu/Macaulay2/'
    aliases = ['macaulay2']
    filenames = ['*.m2']
    tokens = {'root': [('--.*$', Comment.Single), ('-\\*', Comment.Multiline, 'block comment'), ('"', String, 'quote string'), ('///', String, 'slash string'), (words(M2KEYWORDS, prefix='\\b', suffix='\\b'), Keyword), (words(M2DATATYPES, prefix='\\b', suffix='\\b'), Name.Builtin), (words(M2FUNCTIONS, prefix='\\b', suffix='\\b'), Name.Function), (words(M2CONSTANTS, prefix='\\b', suffix='\\b'), Name.Constant), ('\\s+', Text.Whitespace), ('.', Text)], 'block comment': [('[^*-]+', Comment.Multiline), ('\\*-', Comment.Multiline, '#pop'), ('[*-]', Comment.Multiline)], 'quote string': [('[^\\\\"]+', String), ('"', String, '#pop'), ('\\\\"?', String)], 'slash string': [('[^/]+', String), ('(//)+(?!/)', String), ('/(//)+(?!/)', String, '#pop'), ('/', String)]}