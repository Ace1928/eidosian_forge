import re
from pygments.lexer import RegexLexer, include, words, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers._openedge_builtins import OPENEDGEKEYWORDS
class GoodDataCLLexer(RegexLexer):
    """
    Lexer for `GoodData-CL
    <http://github.com/gooddata/GoodData-CL/raw/master/cli/src/main/resources/com/gooddata/processor/COMMANDS.txt>`_
    script files.

    .. versionadded:: 1.4
    """
    name = 'GoodData-CL'
    aliases = ['gooddata-cl']
    filenames = ['*.gdc']
    mimetypes = ['text/x-gooddata-cl']
    flags = re.IGNORECASE
    tokens = {'root': [('#.*', Comment.Single), ('[a-z]\\w*', Name.Function), ('\\(', Punctuation, 'args-list'), (';', Punctuation), ('\\s+', Text)], 'args-list': [('\\)', Punctuation, '#pop'), (',', Punctuation), ('[a-z]\\w*', Name.Variable), ('=', Operator), ('"', String, 'string-literal'), ('[0-9]+(?:\\.[0-9]+)?(?:e[+-]?[0-9]{1,3})?', Number), ('\\s', Text)], 'string-literal': [('\\\\[tnrfbae"\\\\]', String.Escape), ('"', String, '#pop'), ('[^\\\\"]+', String)]}