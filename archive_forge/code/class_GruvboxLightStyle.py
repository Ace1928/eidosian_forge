from pygments.style import Style
from pygments.token import Token, Keyword, Name, Comment, String, Error, \
class GruvboxLightStyle(Style):
    """
    Pygments version of the "gruvbox" Light vim theme.
    """
    background_color = '#fbf1c7'
    highlight_color = '#3c3836'
    styles = {Comment: 'italic #928374', Comment.PreProc: '#427b58', Comment.Special: 'bold italic #3c3836', Keyword: '#9d0006', Operator.Word: '#9d0006', String: '#79740e', String.Escape: '#af3a03', Number: '#8f3f71', Name.Builtin: '#af3a03', Name.Variable: '#076678', Name.Constant: '#8f3f71', Name.Class: '#427b58', Name.Function: '#427b58', Name.Namespace: '#427b58', Name.Exception: '#9d0006', Name.Tag: '#427b58', Name.Attribute: '#b57614', Name.Decorator: '#9d0006', Generic.Heading: 'bold #3c3836', Generic.Subheading: 'underline #3c3836', Generic.Deleted: 'bg:#9d0006 #fbf1c7', Generic.Inserted: 'bg:#79740e #fbf1c7', Generic.Error: '#9d0006', Generic.Emph: 'italic', Generic.Strong: 'bold', Generic.Prompt: '#7c6f64', Generic.Output: '#32302f', Generic.Traceback: '#9d0006', Error: 'bg:#9d0006 #fbf1c7'}