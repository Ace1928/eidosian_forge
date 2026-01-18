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
@validate('column')
def _validate_column(self, proposal):
    column = proposal.value
    if column is None:
        return column
    color = np.asarray(self.color)
    n_columns = color.shape[1]
    if len(column) != n_columns and len(column) != n_columns + 1 and (len(column) != n_columns - 1):
        raise TraitError('column must be an array of size color.shape[1]')
    return column