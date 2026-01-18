import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class KokaLexer(RegexLexer):
    """
    Lexer for the `Koka <http://koka.codeplex.com>`_
    language.

    .. versionadded:: 1.6
    """
    name = 'Koka'
    aliases = ['koka']
    filenames = ['*.kk', '*.kki']
    mimetypes = ['text/x-koka']
    keywords = ['infix', 'infixr', 'infixl', 'type', 'cotype', 'rectype', 'alias', 'struct', 'con', 'fun', 'function', 'val', 'var', 'external', 'if', 'then', 'else', 'elif', 'return', 'match', 'private', 'public', 'private', 'module', 'import', 'as', 'include', 'inline', 'rec', 'try', 'yield', 'enum', 'interface', 'instance']
    typeStartKeywords = ['type', 'cotype', 'rectype', 'alias', 'struct', 'enum']
    typekeywords = ['forall', 'exists', 'some', 'with']
    builtin = ['for', 'while', 'repeat', 'foreach', 'foreach-indexed', 'error', 'catch', 'finally', 'cs', 'js', 'file', 'ref', 'assigned']
    symbols = '[$%&*+@!/\\\\^~=.:\\-?|<>]+'
    sboundary = '(?!' + symbols + ')'
    boundary = '(?![\\w/])'
    tokenType = Name.Attribute
    tokenTypeDef = Name.Class
    tokenConstructor = Generic.Emph
    tokens = {'root': [include('whitespace'), ('::?' + sboundary, tokenType, 'type'), ('(alias)(\\s+)([a-z]\\w*)?', bygroups(Keyword, Text, tokenTypeDef), 'alias-type'), ('(struct)(\\s+)([a-z]\\w*)?', bygroups(Keyword, Text, tokenTypeDef), 'struct-type'), ('(%s)' % '|'.join(typeStartKeywords) + '(\\s+)([a-z]\\w*)?', bygroups(Keyword, Text, tokenTypeDef), 'type'), ('(module)(\\s+)(interface\\s+)?((?:[a-z]\\w*/)*[a-z]\\w*)', bygroups(Keyword, Text, Keyword, Name.Namespace)), ('(import)(\\s+)((?:[a-z]\\w*/)*[a-z]\\w*)(?:(\\s*)(=)(\\s*)((?:qualified\\s*)?)((?:[a-z]\\w*/)*[a-z]\\w*))?', bygroups(Keyword, Text, Name.Namespace, Text, Keyword, Text, Keyword, Name.Namespace)), ('(^(?:(?:public|private)\\s*)?(?:function|fun|val))(\\s+)([a-z]\\w*|\\((?:' + symbols + '|/)\\))', bygroups(Keyword, Text, Name.Function)), ('(^(?:(?:public|private)\\s*)?external)(\\s+)(inline\\s+)?([a-z]\\w*|\\((?:' + symbols + '|/)\\))', bygroups(Keyword, Text, Keyword, Name.Function)), ('(%s)' % '|'.join(typekeywords) + boundary, Keyword.Type), ('(%s)' % '|'.join(keywords) + boundary, Keyword), ('(%s)' % '|'.join(builtin) + boundary, Keyword.Pseudo), ('::?|:=|\\->|[=.]' + sboundary, Keyword), ('((?:[a-z]\\w*/)*)([A-Z]\\w*)', bygroups(Name.Namespace, tokenConstructor)), ('((?:[a-z]\\w*/)*)([a-z]\\w*)', bygroups(Name.Namespace, Name)), ('((?:[a-z]\\w*/)*)(\\((?:' + symbols + '|/)\\))', bygroups(Name.Namespace, Name)), ('_\\w*', Name.Variable), ('@"', String.Double, 'litstring'), (symbols + '|/(?![*/])', Operator), ('`', Operator), ('[{}()\\[\\];,]', Punctuation), ('[0-9]+\\.[0-9]+([eE][\\-+]?[0-9]+)?', Number.Float), ('0[xX][0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ("'", String.Char, 'char'), ('"', String.Double, 'string')], 'alias-type': [('=', Keyword), include('type')], 'struct-type': [('(?=\\((?!,*\\)))', Punctuation, '#pop'), include('type')], 'type': [('[(\\[<]', tokenType, 'type-nested'), include('type-content')], 'type-nested': [('[)\\]>]', tokenType, '#pop'), ('[(\\[<]', tokenType, 'type-nested'), (',', tokenType), ('([a-z]\\w*)(\\s*)(:)(?!:)', bygroups(Name, Text, tokenType)), include('type-content')], 'type-content': [include('whitespace'), ('(%s)' % '|'.join(typekeywords) + boundary, Keyword), ('(?=((%s)' % '|'.join(keywords) + boundary + '))', Keyword, '#pop'), ('[EPHVX]' + boundary, tokenType), ('[a-z][0-9]*(?![\\w/])', tokenType), ('_\\w*', tokenType.Variable), ('((?:[a-z]\\w*/)*)([A-Z]\\w*)', bygroups(Name.Namespace, tokenType)), ('((?:[a-z]\\w*/)*)([a-z]\\w+)', bygroups(Name.Namespace, tokenType)), ('::|->|[.:|]', tokenType), default('#pop')], 'whitespace': [('\\n\\s*#.*$', Comment.Preproc), ('\\s+', Text), ('/\\*', Comment.Multiline, 'comment'), ('//.*$', Comment.Single)], 'comment': [('[^/*]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'litstring': [('[^"]+', String.Double), ('""', String.Escape), ('"', String.Double, '#pop')], 'string': [('[^\\\\"\\n]+', String.Double), include('escape-sequence'), ('["\\n]', String.Double, '#pop')], 'char': [("[^\\\\\\'\\n]+", String.Char), include('escape-sequence'), ("[\\'\\n]", String.Char, '#pop')], 'escape-sequence': [('\\\\[nrt\\\\"\\\']', String.Escape), ('\\\\x[0-9a-fA-F]{2}', String.Escape), ('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\U[0-9a-fA-F]{6}', String.Escape)]}