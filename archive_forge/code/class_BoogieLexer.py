from pygments.lexer import RegexLexer, include, words
from pygments.token import Comment, Operator, Keyword, Name, Number, \
class BoogieLexer(RegexLexer):
    """
    For `Boogie <https://boogie.codeplex.com/>`_ source code.

    .. versionadded:: 2.1
    """
    name = 'Boogie'
    aliases = ['boogie']
    filenames = ['*.bpl']
    tokens = {'root': [('\\n', Whitespace), ('\\s+', Whitespace), ('//[/!](.*?)\\n', Comment.Doc), ('//(.*?)\\n', Comment.Single), ('/\\*', Comment.Multiline, 'comment'), (words(('axiom', 'break', 'call', 'ensures', 'else', 'exists', 'function', 'forall', 'if', 'invariant', 'modifies', 'procedure', 'requires', 'then', 'var', 'while'), suffix='\\b'), Keyword), (words(('const',), suffix='\\b'), Keyword.Reserved), (words(('bool', 'int', 'ref'), suffix='\\b'), Keyword.Type), include('numbers'), ('(>=|<=|:=|!=|==>|&&|\\|\\||[+/\\-=>*<\\[\\]])', Operator), ('([{}():;,.])', Punctuation), ('[a-zA-Z_]\\w*', Name)], 'comment': [('[^*/]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'numbers': [('[0-9]+', Number.Integer)]}