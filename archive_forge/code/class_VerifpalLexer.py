from pygments.lexer import RegexLexer, words, bygroups, default
from pygments.token import Comment, Keyword, Name, String, Punctuation, \
class VerifpalLexer(RegexLexer):
    """
    For Verifpal code.

    .. versionadded:: 2.16
    """
    name = 'Verifpal'
    aliases = ['verifpal']
    filenames = ['*.vp']
    mimetypes = ['text/x-verifpal']
    url = 'https://verifpal.com'
    tokens = {'root': [('//.*$', Comment.Single), ('(principal)( +)(\\w+)( *)(\\[)(.*)$', bygroups(Name.Builtin, Whitespace, String, Whitespace, Punctuation, Whitespace)), ('(attacker)( *)(\\[)( *)(passive|active)( *)(\\])( *)$', bygroups(Name.Builtin, Whitespace, Punctuation, Whitespace, String, Whitespace, Punctuation, Whitespace)), ('(knows)( +)(private|public)( +)', bygroups(Name.Builtin, Whitespace, Keyword.Constant, Whitespace), 'shared'), ('(queries)( +)(\\[)', bygroups(Name.Builtin, Whitespace, Punctuation), 'queries'), ('(\\w+)( +)(->|→)( *)(\\w+)( *)(\\:)', bygroups(String, Whitespace, Punctuation, Whitespace, String, Whitespace, Punctuation), 'shared'), (words(('generates', 'leaks'), suffix='\\b'), Name.Builtin, 'shared'), (words(('phase', 'precondition'), suffix='\\b'), Name.Builtin), ('[\\[\\(\\)\\]\\?:=→^,]', Punctuation), ('->', Punctuation), (words(('password',), suffix='\\b'), Keyword.Constant), (words(('AEAD_DEC', 'AEAD_ENC', 'ASSERT', 'BLIND', 'CONCAT', 'DEC', 'ENC', 'G', 'HASH', 'HKDF', 'MAC', 'PKE_DEC', 'PKE_ENC', 'PW_HASH', 'RINGSIGN', 'RINGSIGNVERIF', 'SHAMIR_JOIN', 'SHAMIR_SPLIT', 'SIGN', 'SIGNVERIF', 'SPLIT', 'UNBLIND', '_', 'nil'), suffix='\\b'), Name.Function), ('\\s+', Whitespace), ('\\w+', Name.Variable)], 'shared': [('[\\^\\[\\],]', Punctuation), (' +', Whitespace), ('\\w+', Name.Variable), default('#pop')], 'queries': [('\\s+', Name.Variable), (words(('confidentiality?', 'authentication?', 'freshness?', 'unlinkability?', 'equivalence?'), suffix='( )'), bygroups(Keyword.Pseudo, Whitespace), 'shared'), default('#pop')]}