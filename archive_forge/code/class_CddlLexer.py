from pygments.lexer import RegexLexer, bygroups, include, words
from pygments.token import Comment, Error, Keyword, Name, Number, Operator, \
class CddlLexer(RegexLexer):
    """
    Lexer for CDDL definitions.

    .. versionadded:: 2.8
    """
    name = 'CDDL'
    url = 'https://datatracker.ietf.org/doc/rfc8610/'
    aliases = ['cddl']
    filenames = ['*.cddl']
    mimetypes = ['text/x-cddl']
    _prelude_types = ['any', 'b64legacy', 'b64url', 'bigfloat', 'bigint', 'bignint', 'biguint', 'bool', 'bstr', 'bytes', 'cbor-any', 'decfrac', 'eb16', 'eb64legacy', 'eb64url', 'encoded-cbor', 'false', 'float', 'float16', 'float16-32', 'float32', 'float32-64', 'float64', 'int', 'integer', 'mime-message', 'nil', 'nint', 'null', 'number', 'regexp', 'tdate', 'text', 'time', 'true', 'tstr', 'uint', 'undefined', 'unsigned', 'uri']
    _controls = ['.and', '.bits', '.cbor', '.cborseq', '.default', '.eq', '.ge', '.gt', '.le', '.lt', '.ne', '.regexp', '.size', '.within']
    _re_id = '[$@A-Z_a-z](?:[\\-\\.]+(?=[$@0-9A-Z_a-z])|[$@0-9A-Z_a-z])*'
    _re_uint = '(?:0b[01]+|0x[0-9a-fA-F]+|[1-9]\\d*|0(?!\\d))'
    _re_int = '-?' + _re_uint
    tokens = {'commentsandwhitespace': [('\\s+', Whitespace), (';.+$', Comment.Single)], 'root': [include('commentsandwhitespace'), ('#(\\d\\.{uint})?'.format(uint=_re_uint), Keyword.Type), ('({uint})?(\\*)({uint})?'.format(uint=_re_uint), bygroups(Number, Operator, Number)), ('\\?|\\+', Operator), ('\\^', Operator), ('(\\.\\.\\.|\\.\\.)', Operator), (words(_controls, suffix='\\b'), Operator.Word), ('&(?=\\s*({groupname}|\\())'.format(groupname=_re_id), Operator), ('~(?=\\s*{})'.format(_re_id), Operator), ('//|/(?!/)', Operator), ('=>|/==|/=|=', Operator), ('[\\[\\]{}\\(\\),<>:]', Punctuation), ("(b64)(')", bygroups(String.Affix, String.Single), 'bstrb64url'), ("(h)(')", bygroups(String.Affix, String.Single), 'bstrh'), ("'", String.Single, 'bstr'), ('({bareword})(\\s*)(:)'.format(bareword=_re_id), bygroups(String, Whitespace, Punctuation)), (words(_prelude_types, prefix='(?![\\-_$@])\\b', suffix='\\b(?![\\-_$@])'), Name.Builtin), (_re_id, Name.Class), ('0b[01]+', Number.Bin), ('0o[0-7]+', Number.Oct), ('0x[0-9a-fA-F]+(\\.[0-9a-fA-F]+)?p[+-]?\\d+', Number.Hex), ('0x[0-9a-fA-F]+', Number.Hex), ('{int}(?=(\\.\\d|e[+-]?\\d))(?:\\.\\d+)?(?:e[+-]?\\d+)?'.format(int=_re_int), Number.Float), (_re_int, Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double)], 'bstrb64url': [("'", String.Single, '#pop'), include('commentsandwhitespace'), ('\\\\.', String.Escape), ('[0-9a-zA-Z\\-_=]+', String.Single), ('.', Error)], 'bstrh': [("'", String.Single, '#pop'), include('commentsandwhitespace'), ('\\\\.', String.Escape), ('[0-9a-fA-F]+', String.Single), ('.', Error)], 'bstr': [("'", String.Single, '#pop'), ('\\\\.', String.Escape), ("[^'\\\\]+", String.Single)]}