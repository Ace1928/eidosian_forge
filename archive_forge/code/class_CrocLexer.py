from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Keyword, Name, String, \
class CrocLexer(RegexLexer):
    """
    For `Croc <http://jfbillingsley.com/croc>`_ source.
    """
    name = 'Croc'
    filenames = ['*.croc']
    aliases = ['croc']
    mimetypes = ['text/x-crocsrc']
    tokens = {'root': [('\\n', Text), ('\\s+', Text), ('//(.*?)\\n', Comment.Single), ('/\\*', Comment.Multiline, 'nestedcomment'), (words(('as', 'assert', 'break', 'case', 'catch', 'class', 'continue', 'default', 'do', 'else', 'finally', 'for', 'foreach', 'function', 'global', 'namespace', 'if', 'import', 'in', 'is', 'local', 'module', 'return', 'scope', 'super', 'switch', 'this', 'throw', 'try', 'vararg', 'while', 'with', 'yield'), suffix='\\b'), Keyword), ('(false|true|null)\\b', Keyword.Constant), ('([0-9][0-9_]*)(?=[.eE])(\\.[0-9][0-9_]*)?([eE][+\\-]?[0-9_]+)?', Number.Float), ('0[bB][01][01_]*', Number.Bin), ('0[xX][0-9a-fA-F][0-9a-fA-F_]*', Number.Hex), ('([0-9][0-9_]*)(?![.eE])', Number.Integer), ('\'(\\\\[\'"\\\\nrt]|\\\\x[0-9a-fA-F]{2}|\\\\[0-9]{1,3}|\\\\u[0-9a-fA-F]{4}|\\\\U[0-9a-fA-F]{8}|.)\'', String.Char), ('@"(""|[^"])*"', String), ('@`(``|[^`])*`', String), ("@'(''|[^'])*'", String), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ('(~=|\\^=|%=|\\*=|==|!=|>>>=|>>>|>>=|>>|>=|<=>|\\?=|-\\>|<<=|<<|<=|\\+\\+|\\+=|--|-=|\\|\\||\\|=|&&|&=|\\.\\.|/=)|[-/.&$@|\\+<>!()\\[\\]{}?,;:=*%^~#\\\\]', Punctuation), ('[a-zA-Z_]\\w*', Name)], 'nestedcomment': [('[^*/]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)]}