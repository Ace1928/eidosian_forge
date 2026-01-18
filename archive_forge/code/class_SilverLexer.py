from pygments.lexer import RegexLexer, include, words
from pygments.token import Comment, Operator, Keyword, Name, Number, \
class SilverLexer(RegexLexer):
    """
    For `Silver <https://bitbucket.org/viperproject/silver>`_ source code.

    .. versionadded:: 2.2
    """
    name = 'Silver'
    aliases = ['silver']
    filenames = ['*.sil', '*.vpr']
    tokens = {'root': [('\\n', Whitespace), ('\\s+', Whitespace), ('//[/!](.*?)\\n', Comment.Doc), ('//(.*?)\\n', Comment.Single), ('/\\*', Comment.Multiline, 'comment'), (words(('result', 'true', 'false', 'null', 'method', 'function', 'predicate', 'program', 'domain', 'axiom', 'var', 'returns', 'field', 'define', 'requires', 'ensures', 'invariant', 'fold', 'unfold', 'inhale', 'exhale', 'new', 'assert', 'assume', 'goto', 'while', 'if', 'elseif', 'else', 'fresh', 'constraining', 'Seq', 'Set', 'Multiset', 'union', 'intersection', 'setminus', 'subset', 'unfolding', 'in', 'old', 'forall', 'exists', 'acc', 'wildcard', 'write', 'none', 'epsilon', 'perm', 'unique', 'apply', 'package', 'folding', 'label', 'forperm'), suffix='\\b'), Keyword), (words(('Int', 'Perm', 'Bool', 'Ref'), suffix='\\b'), Keyword.Type), include('numbers'), ('[!%&*+=|?:<>/\\-\\[\\]]', Operator), ('([{}():;,.])', Punctuation), ('[\\w$]\\w*', Name)], 'comment': [('[^*/]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'numbers': [('[0-9]+', Number.Integer)]}