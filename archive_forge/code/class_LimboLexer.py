import re
from pygments.lexer import RegexLexer, include, bygroups, default
from pygments.token import Punctuation, Text, Comment, Operator, Keyword, \
class LimboLexer(RegexLexer):
    """
    Lexer for `Limbo programming language <http://www.vitanuova.com/inferno/limbo.html>`_

    TODO:
        - maybe implement better var declaration highlighting
        - some simple syntax error highlighting

    .. versionadded:: 2.0
    """
    name = 'Limbo'
    aliases = ['limbo']
    filenames = ['*.b']
    mimetypes = ['text/limbo']
    tokens = {'whitespace': [('^(\\s*)([a-zA-Z_]\\w*:(\\s*)\\n)', bygroups(Text, Name.Label)), ('\\n', Text), ('\\s+', Text), ('#(\\n|(.|\\n)*?[^\\\\]\\n)', Comment.Single)], 'string': [('"', String, '#pop'), ('\\\\([\\\\abfnrtv"\\\']|x[a-fA-F0-9]{2,4}|u[a-fA-F0-9]{4}|U[a-fA-F0-9]{8}|[0-7]{1,3})', String.Escape), ('[^\\\\"\\n]+', String), ('\\\\', String)], 'statements': [('"', String, 'string'), ("'(\\\\.|\\\\[0-7]{1,3}|\\\\x[a-fA-F0-9]{1,2}|[^\\\\\\'\\n])'", String.Char), ('(\\d+\\.\\d*|\\.\\d+|\\d+)[eE][+-]?\\d+', Number.Float), ('(\\d+\\.\\d*|\\.\\d+|\\d+[fF])', Number.Float), ('16r[0-9a-fA-F]+', Number.Hex), ('8r[0-7]+', Number.Oct), ('((([1-3]\\d)|([2-9]))r)?(\\d+)', Number.Integer), ('[()\\[\\],.]', Punctuation), ('[~!%^&*+=|?:<>/-]|(->)|(<-)|(=>)|(::)', Operator), ('(alt|break|case|continue|cyclic|do|else|exitfor|hd|if|implement|import|include|len|load|orpick|return|spawn|tagof|tl|to|while)\\b', Keyword), ('(byte|int|big|real|string|array|chan|list|adt|fn|ref|of|module|self|type)\\b', Keyword.Type), ('(con|iota|nil)\\b', Keyword.Constant), ('[a-zA-Z_]\\w*', Name)], 'statement': [include('whitespace'), include('statements'), ('[{}]', Punctuation), (';', Punctuation, '#pop')], 'root': [include('whitespace'), default('statement')]}

    def analyse_text(text):
        if re.search('^implement \\w+;', text, re.MULTILINE):
            return 0.7