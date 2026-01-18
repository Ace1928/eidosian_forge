from pygments.lexer import RegexLexer, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class MoselLexer(RegexLexer):
    """
    For the Mosel optimization language.

    .. versionadded:: 2.6
    """
    name = 'Mosel'
    aliases = ['mosel']
    filenames = ['*.mos']
    tokens = {'root': [('\\n', Text), ('\\s+', Text.Whitespace), ('!.*?\\n', Comment.Single), ('\\(!(.|\\n)*?!\\)', Comment.Multiline), (words(('and', 'as', 'break', 'case', 'count', 'declarations', 'do', 'dynamic', 'elif', 'else', 'end-', 'end', 'evaluation', 'false', 'forall', 'forward', 'from', 'function', 'hashmap', 'if', 'imports', 'include', 'initialisations', 'initializations', 'inter', 'max', 'min', 'model', 'namespace', 'next', 'not', 'nsgroup', 'nssearch', 'of', 'options', 'or', 'package', 'parameters', 'procedure', 'public', 'prod', 'record', 'repeat', 'requirements', 'return', 'sum', 'then', 'to', 'true', 'union', 'until', 'uses', 'version', 'while', 'with'), prefix='\\b', suffix='\\b'), Keyword.Builtin), (words(('range', 'array', 'set', 'list', 'mpvar', 'mpproblem', 'linctr', 'nlctr', 'integer', 'string', 'real', 'boolean', 'text', 'time', 'date', 'datetime', 'returned', 'Model', 'Mosel', 'counter', 'xmldoc', 'is_sos1', 'is_sos2', 'is_integer', 'is_binary', 'is_continuous', 'is_free', 'is_semcont', 'is_semint', 'is_partint'), prefix='\\b', suffix='\\b'), Keyword.Type), ('(\\+|\\-|\\*|/|=|<=|>=|\\||\\^|<|>|<>|\\.\\.|\\.|:=|::|:|in|mod|div)', Operator), ('[()\\[\\]{},;]+', Punctuation), (words(FUNCTIONS, prefix='\\b', suffix='\\b'), Name.Function), ('(\\d+\\.(?!\\.)\\d*|\\.(?!.)\\d+)([eE][+-]?\\d+)?', Number.Float), ('\\d+([eE][+-]?\\d+)?', Number.Integer), ('[+-]?Infinity', Number.Integer), ('0[xX][0-9a-fA-F]+', Number), ('"', String.Double, 'double_quote'), ("\\'", String.Single, 'single_quote'), ('(\\w+|(\\.(?!\\.)))', Text)], 'single_quote': [("\\'", String.Single, '#pop'), ("[^\\']+", String.Single)], 'double_quote': [('(\\\\"|\\\\[0-7]{1,3}\\D|\\\\[abfnrtv]|\\\\\\\\)', String.Escape), ('\\"', String.Double, '#pop'), ('[^"\\\\]+', String.Double)]}