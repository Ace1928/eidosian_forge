from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
def _setup_segments(self, data, orient):
    ori = ['x', 'y'].index(orient)
    xys = data[['x', 'y']].to_numpy().astype(float)
    segments = np.stack([xys, xys], axis=1)
    segments[:, 0, ori] -= data['width'] / 2
    segments[:, 1, ori] += data['width'] / 2
    return segments