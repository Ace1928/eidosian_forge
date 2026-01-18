from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from seaborn._stats.base import Stat
def _fit_predict(self, data):
    x = data['x']
    y = data['y']
    if x.nunique() <= self.order:
        xx = yy = []
    else:
        p = np.polyfit(x, y, self.order)
        xx = np.linspace(x.min(), x.max(), self.gridsize)
        yy = np.polyval(p, xx)
    return pd.DataFrame(dict(x=xx, y=yy))