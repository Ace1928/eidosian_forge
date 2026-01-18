from pygments.lexer import RegexLexer
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class APLLexer(RegexLexer):
    """
    A simple APL lexer.

    .. versionadded:: 2.0
    """
    name = 'APL'
    aliases = ['apl']
    filenames = ['*.apl']
    tokens = {'root': [('\\s+', Text), (u'[⍝#].*$', Comment.Single), ("\\'((\\'\\')|[^\\'])*\\'", String.Single), ('"(("")|[^"])*"', String.Double), (u'[⋄◇()]', Punctuation), ('[\\[\\];]', String.Regex), (u'⎕[A-Za-zΔ∆⍙][A-Za-zΔ∆⍙_¯0-9]*', Name.Function), (u'[A-Za-zΔ∆⍙][A-Za-zΔ∆⍙_¯0-9]*', Name.Variable), (u'¯?(0[Xx][0-9A-Fa-f]+|[0-9]*\\.?[0-9]+([Ee][+¯]?[0-9]+)?|¯|∞)([Jj]¯?(0[Xx][0-9A-Fa-f]+|[0-9]*\\.?[0-9]+([Ee][+¯]?[0-9]+)?|¯|∞))?', Number), (u'[\\.\\\\/⌿⍀¨⍣⍨⍠⍤∘]', Name.Attribute), (u'[+\\-×÷⌈⌊∣|⍳?*⍟○!⌹<≤=>≥≠≡≢∊⍷∪∩~∨∧⍱⍲⍴,⍪⌽⊖⍉↑↓⊂⊃⌷⍋⍒⊤⊥⍕⍎⊣⊢⍁⍂≈⌸⍯↗]', Operator), (u'⍬', Name.Constant), (u'[⎕⍞]', Name.Variable.Global), (u'[←→]', Keyword.Declaration), (u'[⍺⍵⍶⍹∇:]', Name.Builtin.Pseudo), ('[{}]', Keyword.Type)]}