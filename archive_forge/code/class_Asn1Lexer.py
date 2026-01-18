import re
from pygments.token import  Comment, Operator, Keyword, Name, String, \
from pygments.lexer import RegexLexer, words, bygroups
class Asn1Lexer(RegexLexer):
    """
    Lexer for ASN.1 module definition

    .. versionadded:: 2.16
    """
    flags = re.MULTILINE
    name = 'ASN.1'
    aliases = ['asn1']
    filenames = ['*.asn1']
    url = 'https://www.itu.int/ITU-T/studygroups/com17/languages/X.680-0207.pdf'
    tokens = {'root': [('\\s+', Whitespace), ('--.*$', Comment.Single), ('/\\*', Comment.Multiline, 'comment'), ('\\d+\\.\\d*([eE][-+]?\\d+)?', Number.Float), ('\\d+', Number.Integer), ('&?[a-z][-a-zA-Z0-9]*[a-zA-Z0-9]\\b', Name.Variable), (words(('TRUE', 'FALSE', 'NULL', 'MINUS-INFINITY', 'PLUS-INFINITY', 'MIN', 'MAX'), suffix='\\b'), Keyword.Constant), (word_sequences(TWO_WORDS_TYPES), Keyword.Type), (words(SINGLE_WORD_TYPES, suffix='\\b'), Keyword.Type), ('EXPORTS\\s+ALL\\b', Keyword.Namespace), (words(SINGLE_WORD_NAMESPACE_KEYWORDS, suffix='\\b'), Operator.Namespace), (word_sequences(MULTI_WORDS_DECLARATIONS), Keyword.Declaration), (words(SINGLE_WORDS_DECLARATIONS, suffix='\\b'), Keyword.Declaration), (words(OPERATOR_WORDS, suffix='\\b'), Operator.Word), (words(SINGLE_WORD_KEYWORDS), Keyword), ('&?[A-Z][-a-zA-Z0-9]*[a-zA-Z0-9]\\b', Name.Type), ('(::=|\\.\\.\\.|\\.\\.|\\[\\[|\\]\\]|\\||\\^)', Operator), ('(\\.|,|\\{|\\}|\\(|\\)|\\[|\\])', Punctuation), ('"', String, 'string'), ("('[01 ]*')(B)\\b", bygroups(String, String.Affix)), ("('[0-9A-F ]*')(H)\\b", bygroups(String, String.Affix))], 'comment': [('[^*/]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'string': [('""', String), ('"', String, '#pop'), ('[^"]', String)]}