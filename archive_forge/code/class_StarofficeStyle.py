from pygments.style import Style
from pygments.token import Comment, Error, Literal, Name, Token
class StarofficeStyle(Style):
    """
    Style similar to StarOffice style, also in OpenOffice and LibreOffice.
    """
    styles = {Token: '#000080', Comment: '#696969', Error: '#800000', Literal: '#EE0000', Name: '#008000'}