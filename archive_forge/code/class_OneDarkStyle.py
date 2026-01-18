from pygments.style import Style
from pygments.token import Comment, Keyword, Name, Number, Operator, \
class OneDarkStyle(Style):
    """
    Theme inspired by One Dark Pro for Atom.

    .. versionadded:: 2.11
    """
    background_color = '#282C34'
    styles = {Token: '#ABB2BF', Punctuation: '#ABB2BF', Punctuation.Marker: '#ABB2BF', Keyword: '#C678DD', Keyword.Constant: '#E5C07B', Keyword.Declaration: '#C678DD', Keyword.Namespace: '#C678DD', Keyword.Reserved: '#C678DD', Keyword.Type: '#E5C07B', Name: '#E06C75', Name.Attribute: '#E06C75', Name.Builtin: '#E5C07B', Name.Class: '#E5C07B', Name.Function: 'bold #61AFEF', Name.Function.Magic: 'bold #56B6C2', Name.Other: '#E06C75', Name.Tag: '#E06C75', Name.Decorator: '#61AFEF', Name.Variable.Class: '', String: '#98C379', Number: '#D19A66', Operator: '#56B6C2', Comment: '#7F848E'}