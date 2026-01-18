import re
from pygments.lexer import RegexLexer, DelegatingLexer, \
from pygments.token import Punctuation, Other, Text, Comment, Operator, \
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers.objective import ObjectiveCLexer
from pygments.lexers.d import DLexer
from pygments.lexers.dotnet import CSharpLexer
from pygments.lexers.ruby import RubyLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
class AntlrLexer(RegexLexer):
    """
    Generic `ANTLR`_ Lexer.
    Should not be called directly, instead
    use DelegatingLexer for your target language.

    .. versionadded:: 1.1

    .. _ANTLR: http://www.antlr.org/
    """
    name = 'ANTLR'
    aliases = ['antlr']
    filenames = []
    _id = '[A-Za-z]\\w*'
    _TOKEN_REF = '[A-Z]\\w*'
    _RULE_REF = '[a-z]\\w*'
    _STRING_LITERAL = "\\'(?:\\\\\\\\|\\\\\\'|[^\\']*)\\'"
    _INT = '[0-9]+'
    tokens = {'whitespace': [('\\s+', Whitespace)], 'comments': [('//.*$', Comment), ('/\\*(.|\\n)*?\\*/', Comment)], 'root': [include('whitespace'), include('comments'), ('(lexer|parser|tree)?(\\s*)(grammar\\b)(\\s*)(' + _id + ')(;)', bygroups(Keyword, Whitespace, Keyword, Whitespace, Name.Class, Punctuation)), ('options\\b', Keyword, 'options'), ('tokens\\b', Keyword, 'tokens'), ('(scope)(\\s*)(' + _id + ')(\\s*)(\\{)', bygroups(Keyword, Whitespace, Name.Variable, Whitespace, Punctuation), 'action'), ('(catch|finally)\\b', Keyword, 'exception'), ('(@' + _id + ')(\\s*)(::)?(\\s*)(' + _id + ')(\\s*)(\\{)', bygroups(Name.Label, Whitespace, Punctuation, Whitespace, Name.Label, Whitespace, Punctuation), 'action'), ('((?:protected|private|public|fragment)\\b)?(\\s*)(' + _id + ')(!)?', bygroups(Keyword, Whitespace, Name.Label, Punctuation), ('rule-alts', 'rule-prelims'))], 'exception': [('\\n', Whitespace, '#pop'), ('\\s', Whitespace), include('comments'), ('\\[', Punctuation, 'nested-arg-action'), ('\\{', Punctuation, 'action')], 'rule-prelims': [include('whitespace'), include('comments'), ('returns\\b', Keyword), ('\\[', Punctuation, 'nested-arg-action'), ('\\{', Punctuation, 'action'), ('(throws)(\\s+)(' + _id + ')', bygroups(Keyword, Whitespace, Name.Label)), ('(,)(\\s*)(' + _id + ')', bygroups(Punctuation, Whitespace, Name.Label)), ('options\\b', Keyword, 'options'), ('(scope)(\\s+)(\\{)', bygroups(Keyword, Whitespace, Punctuation), 'action'), ('(scope)(\\s+)(' + _id + ')(\\s*)(;)', bygroups(Keyword, Whitespace, Name.Label, Whitespace, Punctuation)), ('(@' + _id + ')(\\s*)(\\{)', bygroups(Name.Label, Whitespace, Punctuation), 'action'), (':', Punctuation, '#pop')], 'rule-alts': [include('whitespace'), include('comments'), ('options\\b', Keyword, 'options'), (':', Punctuation), ("'(\\\\\\\\|\\\\'|[^'])*'", String), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ('<<([^>]|>[^>])>>', String), ('\\$?[A-Z_]\\w*', Name.Constant), ('\\$?[a-z_]\\w*', Name.Variable), ('(\\+|\\||->|=>|=|\\(|\\)|\\.\\.|\\.|\\?|\\*|\\^|!|\\#|~)', Operator), (',', Punctuation), ('\\[', Punctuation, 'nested-arg-action'), ('\\{', Punctuation, 'action'), (';', Punctuation, '#pop')], 'tokens': [include('whitespace'), include('comments'), ('\\{', Punctuation), ('(' + _TOKEN_REF + ')(\\s*)(=)?(\\s*)(' + _STRING_LITERAL + ')?(\\s*)(;)', bygroups(Name.Label, Whitespace, Punctuation, Whitespace, String, Whitespace, Punctuation)), ('\\}', Punctuation, '#pop')], 'options': [include('whitespace'), include('comments'), ('\\{', Punctuation), ('(' + _id + ')(\\s*)(=)(\\s*)(' + '|'.join((_id, _STRING_LITERAL, _INT, '\\*')) + ')(\\s*)(;)', bygroups(Name.Variable, Whitespace, Punctuation, Whitespace, Text, Whitespace, Punctuation)), ('\\}', Punctuation, '#pop')], 'action': [('(' + '|'.join(('[^${}\\\'"/\\\\]+', '"(\\\\\\\\|\\\\"|[^"])*"', "'(\\\\\\\\|\\\\'|[^'])*'", '//.*$\\n?', '/\\*(.|\\n)*?\\*/', '/(?!\\*)(\\\\\\\\|\\\\/|[^/])*/', '\\\\(?!%)', '/')) + ')+', Other), ('(\\\\)(%)', bygroups(Punctuation, Other)), ('(\\$[a-zA-Z]+)(\\.?)(text|value)?', bygroups(Name.Variable, Punctuation, Name.Property)), ('\\{', Punctuation, '#push'), ('\\}', Punctuation, '#pop')], 'nested-arg-action': [('(' + '|'.join(('[^$\\[\\]\\\'"/]+', '"(\\\\\\\\|\\\\"|[^"])*"', "'(\\\\\\\\|\\\\'|[^'])*'", '//.*$\\n?', '/\\*(.|\\n)*?\\*/', '/(?!\\*)(\\\\\\\\|\\\\/|[^/])*/', '/')) + ')+', Other), ('\\[', Punctuation, '#push'), ('\\]', Punctuation, '#pop'), ('(\\$[a-zA-Z]+)(\\.?)(text|value)?', bygroups(Name.Variable, Punctuation, Name.Property)), ('(\\\\\\\\|\\\\\\]|\\\\\\[|[^\\[\\]])+', Other)]}

    def analyse_text(text):
        return re.search('^\\s*grammar\\s+[a-zA-Z0-9]+\\s*;', text, re.M)