import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class PrologLexer(RegexLexer):
    """
    Lexer for Prolog files.
    """
    name = 'Prolog'
    aliases = ['prolog']
    filenames = ['*.ecl', '*.prolog', '*.pro', '*.pl']
    mimetypes = ['text/x-prolog']
    flags = re.UNICODE | re.MULTILINE
    tokens = {'root': [('^#.*', Comment.Single), ('/\\*', Comment.Multiline, 'nested-comment'), ('%.*', Comment.Single), ("0\\'.", String.Char), ('0b[01]+', Number.Bin), ('0o[0-7]+', Number.Oct), ('0x[0-9a-fA-F]+', Number.Hex), ("\\d\\d?\\'[a-zA-Z0-9]+", Number.Integer), ('(\\d+\\.\\d*|\\d*\\.\\d+)([eE][+-]?[0-9]+)?', Number.Float), ('\\d+', Number.Integer), ('[\\[\\](){}|.,;!]', Punctuation), (':-|-->', Punctuation), ('"(?:\\\\x[0-9a-fA-F]+\\\\|\\\\u[0-9a-fA-F]{4}|\\\\U[0-9a-fA-F]{8}|\\\\[0-7]+\\\\|\\\\["\\nabcefnrstv]|[^\\\\"])*"', String.Double), ("'(?:''|[^'])*'", String.Atom), ('is\\b', Operator), ('(<|>|=<|>=|==|=:=|=|/|//|\\*|\\+|-)(?=\\s|[a-zA-Z0-9\\[])', Operator), ('(mod|div|not)\\b', Operator), ('_', Keyword), ('([a-z]+)(:)', bygroups(Name.Namespace, Punctuation)), (u'([a-zÀ-\u1fff\u3040-\ud7ff\ue000-\uffef][\\w$À-\u1fff\u3040-\ud7ff\ue000-\uffef]*)(\\s*)(:-|-->)', bygroups(Name.Function, Text, Operator)), (u'([a-zÀ-\u1fff\u3040-\ud7ff\ue000-\uffef][\\w$À-\u1fff\u3040-\ud7ff\ue000-\uffef]*)(\\s*)(\\()', bygroups(Name.Function, Text, Punctuation)), (u'[a-zÀ-\u1fff\u3040-\ud7ff\ue000-\uffef][\\w$À-\u1fff\u3040-\ud7ff\ue000-\uffef]*', String.Atom), (u'[#&*+\\-./:<=>?@\\\\^~¡-¿‐-〿]+', String.Atom), ('[A-Z_]\\w*', Name.Variable), (u'\\s+|[\u2000-\u200f\ufff0-\ufffe\uffef]', Text)], 'nested-comment': [('\\*/', Comment.Multiline, '#pop'), ('/\\*', Comment.Multiline, '#push'), ('[^*/]+', Comment.Multiline), ('[*/]', Comment.Multiline)]}

    def analyse_text(text):
        return ':-' in text