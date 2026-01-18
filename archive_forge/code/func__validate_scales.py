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
@validate('scales')
def _validate_scales(self, proposal):
    """
        Validates the `scales` based on the mark's scaled attributes metadata.

        First checks for missing scale and then for 'rtype' compatibility.
        """
    scales = proposal.value
    for name in self.trait_names(scaled=True):
        trait = self.traits()[name]
        if name not in scales:
            if not trait.allow_none:
                raise TraitError('Missing scale for data attribute %s.' % name)
        elif scales[name].rtype != trait.metadata['rtype']:
            raise TraitError('Range type mismatch for scale %s.' % name)
    return scales