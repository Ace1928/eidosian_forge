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
class AntPath(VectorLayer):
    """AntPath class, with VectorLayer as parent class.

    AntPath layer.

    Attributes
    ----------
    locations: list, default []
        Locations through which the ant-path is going.
    use: string, default 'polyline'
        Type of shape to use for the ant-path. Possible values are 'polyline', 'polygon',
        'rectangle' and 'circle'.
    delay: int, default 400
        Add a delay to the animation flux.
    weight: int, default 5
        Weight of the ant-path.
    dash_array: list, default [10, 20]
        The sizes of the animated dashes.
    color: CSS color, default '#0000FF'
        The color of the primary dashes.
    pulse_color: CSS color, default '#FFFFFF'
        The color of the secondary dashes.
    paused: boolean, default False
        Whether the animation is running or not.
    reverse: boolean, default False
        Whether the animation is going backwards or not.
    hardware_accelerated: boolean, default False
        Whether the ant-path uses hardware acceleration.
    radius: int, default 10
        Radius of the circle, if use is set to ‘circle’
    """
    _view_name = Unicode('LeafletAntPathView').tag(sync=True)
    _model_name = Unicode('LeafletAntPathModel').tag(sync=True)
    locations = List().tag(sync=True)
    use = Enum(values=['polyline', 'polygon', 'rectangle', 'circle'], default_value='polyline').tag(sync=True, o=True)
    delay = Int(400).tag(sync=True, o=True)
    weight = Int(5).tag(sync=True, o=True)
    dash_array = List([10, 20]).tag(sync=True, o=True)
    color = Color('#0000FF').tag(sync=True, o=True)
    pulse_color = Color('#FFFFFF').tag(sync=True, o=True)
    paused = Bool(False).tag(sync=True, o=True)
    reverse = Bool(False).tag(sync=True, o=True)
    hardware_accelerated = Bool(False).tag(sync=True, o=True)
    radius = Int(10).tag(sync=True, o=True)