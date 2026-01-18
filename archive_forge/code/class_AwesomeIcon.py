import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
class AwesomeIcon(UILayer):
    """AwesomeIcon class.

    FontAwesome icon used for markers.

    Attributes
    ----------
    name : string, default "home"
        The name of the FontAwesome icon to use.
        See https://fontawesome.com/v4.7.0/icons for available icons.
    marker_color: string, default "blue"
        Color used for the icon background.
    icon_color: string, default "white"
        CSS color used for the FontAwesome icon.
    spin: boolean, default False
        Whether the icon is spinning or not.
    """
    _view_name = Unicode('LeafletAwesomeIconView').tag(sync=True)
    _model_name = Unicode('LeafletAwesomeIconModel').tag(sync=True)
    name = Unicode('home').tag(sync=True)
    marker_color = Enum(values=['white', 'red', 'darkred', 'lightred', 'orange', 'beige', 'green', 'darkgreen', 'lightgreen', 'blue', 'darkblue', 'lightblue', 'purple', 'darkpurple', 'pink', 'cadetblue', 'white', 'gray', 'lightgray', 'black'], default_value='blue').tag(sync=True)
    icon_color = Color('white').tag(sync=True)
    spin = Bool(False).tag(sync=True)