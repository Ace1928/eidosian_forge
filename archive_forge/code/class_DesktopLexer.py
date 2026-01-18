import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, default, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
from pygments.lexers.data import JsonLexer
class DesktopLexer(RegexLexer):
    """
    Lexer for .desktop files.

    .. versionadded:: 2.16
    """
    name = 'Desktop file'
    url = 'https://specifications.freedesktop.org/desktop-entry-spec/desktop-entry-spec-latest.html'
    aliases = ['desktop']
    filenames = ['*.desktop']
    tokens = {'root': [('^[ \\t]*\\n', Whitespace), ('^(#.*)(\\n)', bygroups(Comment.Single, Whitespace)), ('(\\[[^\\]\\n]+\\])(\\n)', bygroups(Keyword, Whitespace)), ('([-A-Za-z0-9]+)(\\[[^\\] \\t=]+\\])?([ \\t]*)(=)([ \\t]*)([^\\n]*)([ \\t\\n]*\\n)', bygroups(Name.Attribute, Name.Namespace, Whitespace, Operator, Whitespace, String, Whitespace))]}

    def analyse_text(text):
        if text.startswith('[Desktop Entry]'):
            return 1.0
        if re.search('^\\[Desktop Entry\\][ \\t]*$', text[:500], re.MULTILINE) is not None:
            return 0.9
        return 0.0