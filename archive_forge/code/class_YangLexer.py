from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Token, Name, String, Comment, Number
class YangLexer(RegexLexer):
    """
    Lexer for YANG, based on RFC7950.

    .. versionadded:: 2.7
    """
    name = 'YANG'
    url = 'https://tools.ietf.org/html/rfc7950/'
    aliases = ['yang']
    filenames = ['*.yang']
    mimetypes = ['application/yang']
    TOP_STMTS_KEYWORDS = ('module', 'submodule')
    MODULE_HEADER_STMT_KEYWORDS = ('belongs-to', 'namespace', 'prefix', 'yang-version')
    META_STMT_KEYWORDS = ('contact', 'description', 'organization', 'reference', 'revision')
    LINKAGE_STMTS_KEYWORDS = ('import', 'include', 'revision-date')
    BODY_STMT_KEYWORDS = ('action', 'argument', 'augment', 'deviation', 'extension', 'feature', 'grouping', 'identity', 'if-feature', 'input', 'notification', 'output', 'rpc', 'typedef')
    DATA_DEF_STMT_KEYWORDS = ('anydata', 'anyxml', 'case', 'choice', 'config', 'container', 'deviate', 'leaf', 'leaf-list', 'list', 'must', 'presence', 'refine', 'uses', 'when')
    TYPE_STMT_KEYWORDS = ('base', 'bit', 'default', 'enum', 'error-app-tag', 'error-message', 'fraction-digits', 'length', 'max-elements', 'min-elements', 'modifier', 'ordered-by', 'path', 'pattern', 'position', 'range', 'require-instance', 'status', 'type', 'units', 'value', 'yin-element')
    LIST_STMT_KEYWORDS = ('key', 'mandatory', 'unique')
    CONSTANTS_KEYWORDS = ('add', 'current', 'delete', 'deprecated', 'false', 'invert-match', 'max', 'min', 'not-supported', 'obsolete', 'replace', 'true', 'unbounded', 'user')
    TYPES = ('binary', 'bits', 'boolean', 'decimal64', 'empty', 'enumeration', 'identityref', 'instance-identifier', 'int16', 'int32', 'int64', 'int8', 'leafref', 'string', 'uint16', 'uint32', 'uint64', 'uint8', 'union')
    suffix_re_pattern = '(?=[^\\w\\-:])'
    tokens = {'comments': [('[^*/]', Comment), ('/\\*', Comment, '#push'), ('\\*/', Comment, '#pop'), ('[*/]', Comment)], 'root': [('\\s+', Text.Whitespace), ('[{};]+', Token.Punctuation), ('(?<![\\-\\w])(and|or|not|\\+|\\.)(?![\\-\\w])', Token.Operator), ('"(?:\\\\"|[^"])*?"', String.Double), ("'(?:\\\\'|[^'])*?'", String.Single), ('/\\*', Comment, 'comments'), ('//.*?$', Comment), ('(?:^|(?<=[\\s{};]))([\\w.-]+)(:)([\\w.-]+)(?=[\\s{};])', bygroups(Name.Namespace, Token.Punctuation, Name.Variable)), ('([0-9]{4}\\-[0-9]{2}\\-[0-9]{2})(?=[\\s{};])', Name.Label), ('([0-9]+\\.[0-9]+)(?=[\\s{};])', Number.Float), ('([0-9]+)(?=[\\s{};])', Number.Integer), (words(TOP_STMTS_KEYWORDS, suffix=suffix_re_pattern), Token.Keyword), (words(MODULE_HEADER_STMT_KEYWORDS, suffix=suffix_re_pattern), Token.Keyword), (words(META_STMT_KEYWORDS, suffix=suffix_re_pattern), Token.Keyword), (words(LINKAGE_STMTS_KEYWORDS, suffix=suffix_re_pattern), Token.Keyword), (words(BODY_STMT_KEYWORDS, suffix=suffix_re_pattern), Token.Keyword), (words(DATA_DEF_STMT_KEYWORDS, suffix=suffix_re_pattern), Token.Keyword), (words(TYPE_STMT_KEYWORDS, suffix=suffix_re_pattern), Token.Keyword), (words(LIST_STMT_KEYWORDS, suffix=suffix_re_pattern), Token.Keyword), (words(TYPES, suffix=suffix_re_pattern), Name.Class), (words(CONSTANTS_KEYWORDS, suffix=suffix_re_pattern), Name.Class), ('[^;{}\\s\\\'"]+', Name.Variable)]}