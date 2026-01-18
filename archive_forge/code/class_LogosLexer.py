import re
from pygments.lexer import RegexLexer, include, bygroups, using, this, words, \
from pygments.token import Text, Keyword, Name, String, Operator, \
from pygments.lexers.c_cpp import CLexer, CppLexer
class LogosLexer(ObjectiveCppLexer):
    """
    For Logos + Objective-C source code with preprocessor directives.

    .. versionadded:: 1.6
    """
    name = 'Logos'
    aliases = ['logos']
    filenames = ['*.x', '*.xi', '*.xm', '*.xmi']
    mimetypes = ['text/x-logos']
    priority = 0.25
    tokens = {'statements': [('(%orig|%log)\\b', Keyword), ('(%c)\\b(\\()(\\s*)([a-zA-Z$_][\\w$]*)(\\s*)(\\))', bygroups(Keyword, Punctuation, Text, Name.Class, Text, Punctuation)), ('(%init)\\b(\\()', bygroups(Keyword, Punctuation), 'logos_init_directive'), ('(%init)(?=\\s*;)', bygroups(Keyword)), ('(%hook|%group)(\\s+)([a-zA-Z$_][\\w$]+)', bygroups(Keyword, Text, Name.Class), '#pop'), ('(%subclass)(\\s+)', bygroups(Keyword, Text), ('#pop', 'logos_classname')), inherit], 'logos_init_directive': [('\\s+', Text), (',', Punctuation, ('logos_init_directive', '#pop')), ('([a-zA-Z$_][\\w$]*)(\\s*)(=)(\\s*)([^);]*)', bygroups(Name.Class, Text, Punctuation, Text, Text)), ('([a-zA-Z$_][\\w$]*)', Name.Class), ('\\)', Punctuation, '#pop')], 'logos_classname': [('([a-zA-Z$_][\\w$]*)(\\s*:\\s*)([a-zA-Z$_][\\w$]*)?', bygroups(Name.Class, Text, Name.Class), '#pop'), ('([a-zA-Z$_][\\w$]*)', Name.Class, '#pop')], 'root': [('(%subclass)(\\s+)', bygroups(Keyword, Text), 'logos_classname'), ('(%hook|%group)(\\s+)([a-zA-Z$_][\\w$]+)', bygroups(Keyword, Text, Name.Class)), ('(%config)(\\s*\\(\\s*)(\\w+)(\\s*=\\s*)(.*?)(\\s*\\)\\s*)', bygroups(Keyword, Text, Name.Variable, Text, String, Text)), ('(%ctor)(\\s*)(\\{)', bygroups(Keyword, Text, Punctuation), 'function'), ('(%new)(\\s*)(\\()(\\s*.*?\\s*)(\\))', bygroups(Keyword, Text, Keyword, String, Keyword)), ('(\\s*)(%end)(\\s*)', bygroups(Text, Keyword, Text)), inherit]}
    _logos_keywords = re.compile('%(?:hook|ctor|init|c\\()')

    def analyse_text(text):
        if LogosLexer._logos_keywords.search(text):
            return 1.0
        return 0