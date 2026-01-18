import re
from pygments.lexer import RegexLexer, bygroups, default
from pygments.token import Keyword, Punctuation, String, Number, Operator, Generic, \
class SparqlLexer(RegexLexer):
    """
    Lexer for `SPARQL <http://www.w3.org/TR/rdf-sparql-query/>`_ query language.

    .. versionadded:: 2.0
    """
    name = 'SPARQL'
    aliases = ['sparql']
    filenames = ['*.rq', '*.sparql']
    mimetypes = ['application/sparql-query']
    PN_CHARS_BASE_GRP = u'a-zA-ZÀ-ÖØ-öø-˿Ͱ-ͽͿ-\u1fff\u200c-\u200d⁰-\u218fⰀ-\u2fef、-\ud7ff豈-﷏ﷰ-�'
    PN_CHARS_U_GRP = PN_CHARS_BASE_GRP + '_'
    PN_CHARS_GRP = PN_CHARS_U_GRP + '\\-' + '0-9' + u'·' + u'̀-ͯ' + u'‿-⁀'
    HEX_GRP = '0-9A-Fa-f'
    PN_LOCAL_ESC_CHARS_GRP = ' _~.\\-!$&"()*+,;=/?#@%'
    PN_CHARS_BASE = '[' + PN_CHARS_BASE_GRP + ']'
    PN_CHARS_U = '[' + PN_CHARS_U_GRP + ']'
    PN_CHARS = '[' + PN_CHARS_GRP + ']'
    HEX = '[' + HEX_GRP + ']'
    PN_LOCAL_ESC_CHARS = '[' + PN_LOCAL_ESC_CHARS_GRP + ']'
    IRIREF = '<(?:[^<>"{}|^`\\\\\\x00-\\x20])*>'
    BLANK_NODE_LABEL = '_:[0-9' + PN_CHARS_U_GRP + '](?:[' + PN_CHARS_GRP + '.]*' + PN_CHARS + ')?'
    PN_PREFIX = PN_CHARS_BASE + '(?:[' + PN_CHARS_GRP + '.]*' + PN_CHARS + ')?'
    VARNAME = u'[0-9' + PN_CHARS_U_GRP + '][' + PN_CHARS_U_GRP + u'0-9·̀-ͯ‿-⁀]*'
    PERCENT = '%' + HEX + HEX
    PN_LOCAL_ESC = '\\\\' + PN_LOCAL_ESC_CHARS
    PLX = '(?:' + PERCENT + ')|(?:' + PN_LOCAL_ESC + ')'
    PN_LOCAL = '(?:[' + PN_CHARS_U_GRP + ':0-9' + ']|' + PLX + ')' + '(?:(?:[' + PN_CHARS_GRP + '.:]|' + PLX + ')*(?:[' + PN_CHARS_GRP + ':]|' + PLX + '))?'
    EXPONENT = '[eE][+-]?\\d+'
    tokens = {'root': [('\\s+', Text), ('((?i)select|construct|describe|ask|where|filter|group\\s+by|minus|distinct|reduced|from\\s+named|from|order\\s+by|desc|asc|limit|offset|bindings|load|clear|drop|create|add|move|copy|insert\\s+data|delete\\s+data|delete\\s+where|delete|insert|using\\s+named|using|graph|default|named|all|optional|service|silent|bind|union|not\\s+in|in|as|having|to|prefix|base)\\b', Keyword), ('(a)\\b', Keyword), ('(' + IRIREF + ')', Name.Label), ('(' + BLANK_NODE_LABEL + ')', Name.Label), ('[?$]' + VARNAME, Name.Variable), ('(' + PN_PREFIX + ')?(\\:)(' + PN_LOCAL + ')?', bygroups(Name.Namespace, Punctuation, Name.Tag)), ('((?i)str|lang|langmatches|datatype|bound|iri|uri|bnode|rand|abs|ceil|floor|round|concat|strlen|ucase|lcase|encode_for_uri|contains|strstarts|strends|strbefore|strafter|year|month|day|hours|minutes|seconds|timezone|tz|now|md5|sha1|sha256|sha384|sha512|coalesce|if|strlang|strdt|sameterm|isiri|isuri|isblank|isliteral|isnumeric|regex|substr|replace|exists|not\\s+exists|count|sum|min|max|avg|sample|group_concat|separator)\\b', Name.Function), ('(true|false)', Keyword.Constant), ('[+\\-]?(\\d+\\.\\d*' + EXPONENT + '|\\.?\\d+' + EXPONENT + ')', Number.Float), ('[+\\-]?(\\d+\\.\\d*|\\.\\d+)', Number.Float), ('[+\\-]?\\d+', Number.Integer), ('(\\|\\||&&|=|\\*|\\-|\\+|/|!=|<=|>=|!|<|>)', Operator), ('[(){}.;,:^\\[\\]]', Punctuation), ('#[^\\n]*', Comment), ('"""', String, 'triple-double-quoted-string'), ('"', String, 'single-double-quoted-string'), ("'''", String, 'triple-single-quoted-string'), ("'", String, 'single-single-quoted-string')], 'triple-double-quoted-string': [('"""', String, 'end-of-string'), ('[^\\\\]+', String), ('\\\\', String, 'string-escape')], 'single-double-quoted-string': [('"', String, 'end-of-string'), ('[^"\\\\\\n]+', String), ('\\\\', String, 'string-escape')], 'triple-single-quoted-string': [("'''", String, 'end-of-string'), ('[^\\\\]+', String), ('\\\\', String.Escape, 'string-escape')], 'single-single-quoted-string': [("'", String, 'end-of-string'), ("[^'\\\\\\n]+", String), ('\\\\', String, 'string-escape')], 'string-escape': [('u' + HEX + '{4}', String.Escape, '#pop'), ('U' + HEX + '{8}', String.Escape, '#pop'), ('.', String.Escape, '#pop')], 'end-of-string': [('(@)([a-zA-Z]+(?:-[a-zA-Z0-9]+)*)', bygroups(Operator, Name.Function), '#pop:2'), ('\\^\\^', Operator, '#pop:2'), default('#pop:2')]}