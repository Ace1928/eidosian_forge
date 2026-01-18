from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
def _standardize_coordinate_parameters(self, data, orient):
    other = {'x': 'y', 'y': 'x'}[orient]
    if not set(data.columns) & {f'{other}min', f'{other}max'}:
        agg = {f'{other}min': (other, 'min'), f'{other}max': (other, 'max')}
        data = data.groupby(orient).agg(**agg).reset_index()
    return data