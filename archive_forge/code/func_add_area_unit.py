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
def add_area_unit(self, name, factor, decimals=0):
    """Add a custom area unit.

        Parameters
        ----------
        name: str
            The name for your custom unit.
        factor: float
            Factor to apply when converting to this unit. Area in sqmeters
            will be multiplied by this factor.
        decimals: int, default 0
            Number of decimals to round results when using this unit.
        """
    self._area_units.append(name)
    self._add_unit(name, factor, decimals)