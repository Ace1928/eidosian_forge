import pathlib
import param
from ...io.state import state
from ...theme import THEMES, DefaultTheme
from ...theme.fast import Design, Fast
from ..base import BasicTemplate
from ..react import ReactTemplate
class FastGridBaseTemplate(FastBaseTemplate, ReactTemplate):
    """
    Combines the FastTemplate and the React template.
    """
    _resources = dict(js=ReactTemplate._resources['js'])
    __abstract = True