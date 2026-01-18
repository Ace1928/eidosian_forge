import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
def _rx_indent(level):
    tab_width = 8
    if tab_width == 2:
        space_repeat = '+'
    else:
        space_repeat = '{1,%d}' % (tab_width - 1)
    if level == 1:
        level_repeat = ''
    else:
        level_repeat = '{%s}' % level
    return '(?:\\t| %s\\t| {%s})%s.*\\n' % (space_repeat, tab_width, level_repeat)