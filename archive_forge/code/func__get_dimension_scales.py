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
def _get_dimension_scales(self, dimension, preserve_domain=False):
    """
        Return the list of scales corresponding to a given dimension.

        The preserve_domain optional argument specifies whether one should
        filter out the scales for which preserve_domain is set to True.
        """
    if preserve_domain:
        return [self.scales[k] for k in self.scales if k in self.scales_metadata and self.scales_metadata[k].get('dimension') == dimension and (not self.preserve_domain.get(k))]
    else:
        return [self.scales[k] for k in self.scales if k in self.scales_metadata and self.scales_metadata[k].get('dimension') == dimension]