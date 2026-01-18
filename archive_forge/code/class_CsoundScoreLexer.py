import re
from pygments.lexer import RegexLexer, bygroups, default, include, using, words
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, \
from pygments.lexers._csound_builtins import OPCODES
from pygments.lexers.html import HtmlLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.scripting import LuaLexer
class CsoundScoreLexer(CsoundLexer):
    """
    For `Csound <http://csound.github.io>`_ scores.

    .. versionadded:: 2.1
    """
    name = 'Csound Score'
    aliases = ['csound-score', 'csound-sco']
    filenames = ['*.sco']
    tokens = {'partial statement': [include('preprocessor directives'), ('\\d+e[+-]?\\d+|(\\d+\\.\\d*|\\d*\\.\\d+)(e[+-]?\\d+)?', Number.Float), ('0[xX][a-fA-F0-9]+', Number.Hex), ('\\d+', Number.Integer), ('"', String, 'single-line string'), ('[+\\-*/%^!=<>|&#~.]', Operator), ('[]()[]', Punctuation), ('\\w+', Comment.Preproc)], 'statement': [include('whitespace or macro call'), newline + ('#pop',), include('partial statement')], 'root': [newline, include('whitespace or macro call'), ('[{}]', Punctuation, 'statement'), ('[abefimq-tv-z]|[nN][pP]?', Keyword, 'statement')], 'single-line string': [('"', String, '#pop'), ('[^\\\\"]+', String)]}