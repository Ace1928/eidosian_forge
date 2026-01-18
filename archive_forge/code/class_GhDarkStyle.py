from pygments.style import Style
from pygments.token import Keyword, Name, Comment, Error, Number, Operator, \
class GhDarkStyle(Style):
    """
    Github's Dark-Colorscheme based theme for Pygments
    """
    background_color = BG_DEFAULT
    highlight_color = GRAY_4
    line_number_special_color = FG_DEFAULT
    line_number_special_background_color = FG_SUBTLE
    line_number_color = GRAY_4
    line_number_background_color = BG_DEFAULT
    styles = {Token: FG_DEFAULT, Error: DANGER_FG, Keyword: RED_3, Keyword.Constant: BLUE_2, Keyword.Pseudo: BLUE_2, Name: FG_DEFAULT, Name.Class: 'bold ' + ORANGE_3, Name.Constant: 'bold ' + BLUE_2, Name.Decorator: 'bold ' + PURPLE_2, Name.Entity: ORANGE_2, Name.Exception: 'bold ' + ORANGE_3, Name.Function: 'bold ' + PURPLE_2, Name.Label: 'bold ' + BLUE_2, Name.Namespace: RED_3, Name.Property: BLUE_2, Name.Tag: GREEN_1, Name.Variable: BLUE_2, Literal: BLUE_1, Literal.Date: BLUE_2, String: BLUE_1, String.Affix: BLUE_2, String.Delimiter: BLUE_2, String.Escape: BLUE_2, String.Heredoc: BLUE_2, String.Regex: BLUE_2, Number: BLUE_1, Comment: 'italic ' + GRAY_3, Comment.Preproc: 'bold ' + GRAY_3, Comment.Special: 'bold italic ' + GRAY_3, Operator: 'bold ' + RED_3, Generic: FG_DEFAULT, Generic.Deleted: f'bg:{RED_9} {RED_2}', Generic.Emph: 'italic', Generic.Error: RED_2, Generic.Heading: 'bold ' + BLUE_2, Generic.Inserted: f'bg:{GREEN_7} {GREEN_2}', Generic.Output: GRAY_3, Generic.Prompt: GRAY_3, Generic.Strong: 'bold', Generic.EmphStrong: 'bold italic', Generic.Subheading: BLUE_2, Generic.Traceback: RED_3, Generic.Underline: 'underline', Text.Whitespace: FG_SUBTLE}