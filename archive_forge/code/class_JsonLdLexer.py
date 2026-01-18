import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
class JsonLdLexer(JsonLexer):
    """
    For `JSON-LD <http://json-ld.org/>`_ linked data.

    .. versionadded:: 2.0
    """
    name = 'JSON-LD'
    aliases = ['jsonld', 'json-ld']
    filenames = ['*.jsonld']
    mimetypes = ['application/ld+json']
    tokens = {'objectvalue': [('"@(context|id|value|language|type|container|list|set|reverse|index|base|vocab|graph)"', Name.Decorator, 'objectattribute'), inherit]}