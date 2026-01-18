import re
from pygments.lexer import RegexLexer, bygroups, default
from pygments.token import Keyword, Punctuation, String, Number, Operator, \
class ShExCLexer(RegexLexer):
    """
    Lexer for `ShExC <https://shex.io/shex-semantics/#shexc>`_ shape expressions language syntax.
    """
    name = 'ShExC'
    aliases = ['shexc', 'shex']
    filenames = ['*.shex']
    mimetypes = ['text/shex']
    PN_CHARS_BASE_GRP = 'a-zA-ZÀ-ÖØ-öø-˿Ͱ-ͽͿ-\u1fff\u200c-\u200d⁰-\u218fⰀ-\u2fef、-\ud7ff豈-﷏ﷰ-�'
    PN_CHARS_U_GRP = PN_CHARS_BASE_GRP + '_'
    PN_CHARS_GRP = PN_CHARS_U_GRP + '\\-' + '0-9' + '·' + '̀-ͯ' + '‿-⁀'
    HEX_GRP = '0-9A-Fa-f'
    PN_LOCAL_ESC_CHARS_GRP = "_~.\\-!$&'()*+,;=/?#@%"
    PN_CHARS_BASE = '[' + PN_CHARS_BASE_GRP + ']'
    PN_CHARS_U = '[' + PN_CHARS_U_GRP + ']'
    PN_CHARS = '[' + PN_CHARS_GRP + ']'
    HEX = '[' + HEX_GRP + ']'
    PN_LOCAL_ESC_CHARS = '[' + PN_LOCAL_ESC_CHARS_GRP + ']'
    UCHAR_NO_BACKSLASH = '(?:u' + HEX + '{4}|U' + HEX + '{8})'
    UCHAR = '\\\\' + UCHAR_NO_BACKSLASH
    IRIREF = '<(?:[^\\x00-\\x20<>"{}|^`\\\\]|' + UCHAR + ')*>'
    BLANK_NODE_LABEL = '_:[0-9' + PN_CHARS_U_GRP + '](?:[' + PN_CHARS_GRP + '.]*' + PN_CHARS + ')?'
    PN_PREFIX = PN_CHARS_BASE + '(?:[' + PN_CHARS_GRP + '.]*' + PN_CHARS + ')?'
    PERCENT = '%' + HEX + HEX
    PN_LOCAL_ESC = '\\\\' + PN_LOCAL_ESC_CHARS
    PLX = '(?:' + PERCENT + ')|(?:' + PN_LOCAL_ESC + ')'
    PN_LOCAL = '(?:[' + PN_CHARS_U_GRP + ':0-9' + ']|' + PLX + ')' + '(?:(?:[' + PN_CHARS_GRP + '.:]|' + PLX + ')*(?:[' + PN_CHARS_GRP + ':]|' + PLX + '))?'
    EXPONENT = '[eE][+-]?\\d+'
    tokens = {'root': [('\\s+', Text), ('(?i)(base|prefix|start|external|literal|iri|bnode|nonliteral|length|minlength|maxlength|mininclusive|minexclusive|maxinclusive|maxexclusive|totaldigits|fractiondigits|closed|extra)\\b', Keyword), ('(a)\\b', Keyword), ('(' + IRIREF + ')', Name.Label), ('(' + BLANK_NODE_LABEL + ')', Name.Label), ('(' + PN_PREFIX + ')?(\\:)(' + PN_LOCAL + ')?', bygroups(Name.Namespace, Punctuation, Name.Tag)), ('(true|false)', Keyword.Constant), ('[+\\-]?(\\d+\\.\\d*' + EXPONENT + '|\\.?\\d+' + EXPONENT + ')', Number.Float), ('[+\\-]?(\\d+\\.\\d*|\\.\\d+)', Number.Float), ('[+\\-]?\\d+', Number.Integer), ('[@|$&=*+?^\\-~]', Operator), ('(?i)(and|or|not)\\b', Operator.Word), ('[(){}.;,:^\\[\\]]', Punctuation), ('#[^\\n]*', Comment), ('"""', String, 'triple-double-quoted-string'), ('"', String, 'single-double-quoted-string'), ("'''", String, 'triple-single-quoted-string'), ("'", String, 'single-single-quoted-string')], 'triple-double-quoted-string': [('"""', String, 'end-of-string'), ('[^\\\\]+', String), ('\\\\', String, 'string-escape')], 'single-double-quoted-string': [('"', String, 'end-of-string'), ('[^"\\\\\\n]+', String), ('\\\\', String, 'string-escape')], 'triple-single-quoted-string': [("'''", String, 'end-of-string'), ('[^\\\\]+', String), ('\\\\', String.Escape, 'string-escape')], 'single-single-quoted-string': [("'", String, 'end-of-string'), ("[^'\\\\\\n]+", String), ('\\\\', String, 'string-escape')], 'string-escape': [(UCHAR_NO_BACKSLASH, String.Escape, '#pop'), ('.', String.Escape, '#pop')], 'end-of-string': [('(@)([a-zA-Z]+(?:-[a-zA-Z0-9]+)*)', bygroups(Operator, Name.Function), '#pop:2'), ('\\^\\^', Operator, '#pop:2'), default('#pop:2')]}