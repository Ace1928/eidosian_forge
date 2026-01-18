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
@validate('panes')
def _validate_panes(self, proposal):
    """Validate panes.
        """
    error_msg = "Panes should look like: {'pane_name': {'zIndex': 650, 'pointerEvents': 'none'}, ...}"
    for k1, v1 in proposal.value.items():
        if not isinstance(k1, str) or not isinstance(v1, dict):
            raise PaneException(error_msg)
        for k2, v2 in v1.items():
            if not isinstance(k2, str) or not isinstance(v2, (str, int, float)):
                raise PaneException(error_msg)
    return proposal.value