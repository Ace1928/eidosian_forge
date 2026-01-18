from pygments.style import Style
from pygments.token import Comment, Error, Generic, Keyword, Name, Number, \
class SolarizedDarkStyle(Style):
    """
    The solarized style, dark.
    """
    styles = make_style(DARK_COLORS)
    background_color = DARK_COLORS['base03']
    highlight_color = DARK_COLORS['base02']
    line_number_color = DARK_COLORS['base01']
    line_number_background_color = DARK_COLORS['base02']