import re
from pygments.lexer import RegexLexer, bygroups, default, words, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CbmBasicV2Lexer(RegexLexer):
    """
    For CBM BASIC V2 sources.

    .. versionadded:: 1.6
    """
    name = 'CBM BASIC V2'
    aliases = ['cbmbas']
    filenames = ['*.bas']
    flags = re.IGNORECASE
    tokens = {'root': [('rem.*\\n', Comment.Single), ('\\s+', Text), ('new|run|end|for|to|next|step|go(to|sub)?|on|return|stop|cont|if|then|input#?|read|wait|load|save|verify|poke|sys|print#?|list|clr|cmd|open|close|get#?', Keyword.Reserved), ('data|restore|dim|let|def|fn', Keyword.Declaration), ('tab|spc|sgn|int|abs|usr|fre|pos|sqr|rnd|log|exp|cos|sin|tan|atn|peek|len|val|asc|(str|chr|left|right|mid)\\$', Name.Builtin), ('[-+*/^<>=]', Operator), ('not|and|or', Operator.Word), ('"[^"\\n]*.', String), ('\\d+|[-+]?\\d*\\.\\d*(e[-+]?\\d+)?', Number.Float), ('[(),:;]', Punctuation), ('\\w+[$%]?', Name)]}

    def analyse_text(self, text):
        if re.match('\\d+', text):
            return 0.2