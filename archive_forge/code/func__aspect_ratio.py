from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from .._utils import cross_join, match
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from .strips import Strips
def _aspect_ratio(self) -> Optional[float]:
    """
        Return the aspect_ratio
        """
    aspect_ratio = self.theme.getp('aspect_ratio')
    if aspect_ratio == 'auto':
        if not self.free['x'] and (not self.free['y']):
            aspect_ratio = self.coordinates.aspect(self.layout.panel_params[0])
        else:
            aspect_ratio = None
    return aspect_ratio