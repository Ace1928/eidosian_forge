from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
def _get_verts(self, data, orient):
    dv = {'x': 'y', 'y': 'x'}[orient]
    data = data.sort_values(orient, kind='mergesort')
    verts = np.concatenate([data[[orient, f'{dv}min']].to_numpy(), data[[orient, f'{dv}max']].to_numpy()[::-1]])
    if orient == 'y':
        verts = verts[:, ::-1]
    return verts