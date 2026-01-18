from bokeh.core.enums import ButtonType
from bokeh.core.properties import (
from bokeh.models.ui import Tooltip
from bokeh.models.ui.icons import Icon
from bokeh.models.widgets import (
from .layout import HTMLBox
class CustomSelect(Select):
    """ Custom widget that extends the base Bokeh Select
    by adding a parameter to disable one or more options.

    """
    disabled_options = List(Any, default=[], help='\n    List of options to disable.\n    ')
    size = Int(default=1)