import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
class JsonBareObjectLexer(JsonLexer):
    """
    For JSON data structures (with missing object curly braces).

    .. versionadded:: 2.2
    """
    name = 'JSONBareObject'
    aliases = ['json-object']
    filenames = []
    mimetypes = ['application/json-object']
    tokens = {'root': [('\\}', Error), include('objectvalue')], 'objectattribute': [('\\}', Error), inherit]}