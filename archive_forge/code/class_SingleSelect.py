from bokeh.core.enums import ButtonType
from bokeh.core.properties import (
from bokeh.models.ui import Tooltip
from bokeh.models.ui.icons import Icon
from bokeh.models.widgets import (
from .layout import HTMLBox
class SingleSelect(InputWidget):
    """ Single-select widget.

    """
    disabled_options = List(Any, default=[], help='\n    List of options to disable.\n    ')
    options = List(Either(String, Tuple(String, String)), help='\n    Available selection options. Options may be provided either as a list of\n    possible string values, or as a list of tuples, each of the form\n    ``(value, label)``. In the latter case, the visible widget text for each\n    value will be corresponding given label.\n    ')
    size = Int(default=4, help="\n    The number of visible options in the dropdown list. (This uses the\n    ``select`` HTML element's ``size`` attribute. Some browsers might not\n    show less than 3 options.)\n    ")
    value = Nullable(String, help='Initial or selected value.')