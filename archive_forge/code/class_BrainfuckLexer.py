from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class BrainfuckLexer(RegexLexer):
    """
    Lexer for the esoteric `BrainFuck <http://www.muppetlabs.com/~breadbox/bf/>`_
    language.
    """
    name = 'Brainfuck'
    aliases = ['brainfuck', 'bf']
    filenames = ['*.bf', '*.b']
    mimetypes = ['application/x-brainfuck']
    tokens = {'common': [('[.,]+', Name.Tag), ('[+-]+', Name.Builtin), ('[<>]+', Name.Variable), ('[^.,+\\-<>\\[\\]]+', Comment)], 'root': [('\\[', Keyword, 'loop'), ('\\]', Error), include('common')], 'loop': [('\\[', Keyword, '#push'), ('\\]', Keyword, '#pop'), include('common')]}