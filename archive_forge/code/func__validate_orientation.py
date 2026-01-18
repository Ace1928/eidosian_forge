import os
import json
from warnings import warn
import ipywidgets as widgets
from ipywidgets import (Widget, DOMWidget, CallbackDispatcher,
from traitlets import (Int, Unicode, List, Enum, Dict, Bool, Float,
from traittypes import Array
from numpy import histogram
import numpy as np
from .scales import Scale, OrdinalScale, LinearScale
from .traits import (Date, array_serialization,
from ._version import __frontend_version__
from .colorschemes import CATEGORY10
@validate('orientation')
def _validate_orientation(self, proposal):
    value = proposal['value']
    x_orient = 'horizontal' if value == 'vertical' else 'vertical'
    self.scales_metadata = {'x': {'orientation': x_orient, 'dimension': 'x'}, 'y': {'orientation': value, 'dimension': 'y'}}
    return value