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
class AttributionControl(Control):
    """AttributionControl class.

    A control which contains the layers attribution.
    """
    _view_name = Unicode('LeafletAttributionControlView').tag(sync=True)
    _model_name = Unicode('LeafletAttributionControlModel').tag(sync=True)
    prefix = Unicode('ipyleaflet').tag(sync=True, o=True)