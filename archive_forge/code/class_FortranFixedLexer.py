import re
from pygments.lexer import RegexLexer, bygroups, include, words, using, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class FortranFixedLexer(RegexLexer):
    """
    Lexer for fixed format Fortran.

    .. versionadded:: 2.1
    """
    name = 'FortranFixed'
    aliases = ['fortranfixed']
    filenames = ['*.f', '*.F']
    flags = re.IGNORECASE

    def _lex_fortran(self, match, ctx=None):
        """Lex a line just as free form fortran without line break."""
        lexer = FortranLexer()
        text = match.group(0) + '\n'
        for index, token, value in lexer.get_tokens_unprocessed(text):
            value = value.replace('\n', '')
            if value != '':
                yield (index, token, value)
    tokens = {'root': [('[C*].*\\n', Comment), ('#.*\\n', Comment.Preproc), (' {0,4}!.*\\n', Comment), ('(.{5})', Name.Label, 'cont-char'), ('.*\\n', using(FortranLexer))], 'cont-char': [(' ', Text, 'code'), ('0', Comment, 'code'), ('.', Generic.Strong, 'code')], 'code': [('(.{66})(.*)(\\n)', bygroups(_lex_fortran, Comment, Text), 'root'), ('(.*)(\\n)', bygroups(_lex_fortran, Text), 'root'), default('root')]}