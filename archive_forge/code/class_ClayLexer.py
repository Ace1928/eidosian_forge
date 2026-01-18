import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class ClayLexer(RegexLexer):
    """
    For `Clay <http://claylabs.com/clay/>`_ source.

    .. versionadded:: 2.0
    """
    name = 'Clay'
    filenames = ['*.clay']
    aliases = ['clay']
    mimetypes = ['text/x-clay']
    tokens = {'root': [('\\s', Text), ('//.*?$', Comment.Singleline), ('/(\\\\\\n)?[*](.|\\n)*?[*](\\\\\\n)?/', Comment.Multiline), ('\\b(public|private|import|as|record|variant|instance|define|overload|default|external|alias|rvalue|ref|forward|inline|noinline|forceinline|enum|var|and|or|not|if|else|goto|return|while|switch|case|break|continue|for|in|true|false|try|catch|throw|finally|onerror|staticassert|eval|when|newtype|__FILE__|__LINE__|__COLUMN__|__ARG__)\\b', Keyword), ('[~!%^&*+=|:<>/-]', Operator), ('[#(){}\\[\\],;.]', Punctuation), ('0x[0-9a-fA-F]+[LlUu]*', Number.Hex), ('\\d+[LlUu]*', Number.Integer), ('\\b(true|false)\\b', Name.Builtin), ('(?i)[a-z_?][\\w?]*', Name), ('"""', String, 'tdqs'), ('"', String, 'dqs')], 'strings': [('(?i)\\\\(x[0-9a-f]{2}|.)', String.Escape), ('.', String)], 'nl': [('\\n', String)], 'dqs': [('"', String, '#pop'), include('strings')], 'tdqs': [('"""', String, '#pop'), include('strings'), include('nl')]}