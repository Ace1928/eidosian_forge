from pygments.style import Style
from pygments.token import Token, Keyword, Name, Comment, String, Error, \
class StataDarkStyle(Style):
    background_color = '#232629'
    highlight_color = '#49483e'
    styles = {Token: '#cccccc', Whitespace: '#bbbbbb', Error: 'bg:#e3d2d2 #a61717', String: '#51cc99', Number: '#4FB8CC', Operator: '', Name.Function: '#6a6aff', Name.Other: '#e2828e', Keyword: 'bold #7686bb', Keyword.Constant: '', Comment: 'italic #777777', Name.Variable: 'bold #7AB4DB', Name.Variable.Global: 'bold #BE646C', Generic.Prompt: '#ffffff'}