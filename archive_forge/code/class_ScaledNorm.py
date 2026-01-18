from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure
from seaborn.utils import _version_predates
class ScaledNorm(mpl.colors.Normalize):

    def __call__(self, value, clip=None):
        value, is_scalar = self.process_value(value)
        self.autoscale_None(value)
        if self.vmin > self.vmax:
            raise ValueError('vmin must be less or equal to vmax')
        if self.vmin == self.vmax:
            return np.full_like(value, 0)
        if clip is None:
            clip = self.clip
        if clip:
            value = np.clip(value, self.vmin, self.vmax)
        t_value = self.transform(value).reshape(np.shape(value))
        t_vmin, t_vmax = self.transform([self.vmin, self.vmax])
        if not np.isfinite([t_vmin, t_vmax]).all():
            raise ValueError('Invalid vmin or vmax')
        t_value -= t_vmin
        t_value /= t_vmax - t_vmin
        t_value = np.ma.masked_invalid(t_value, copy=False)
        return t_value[0] if is_scalar else t_value