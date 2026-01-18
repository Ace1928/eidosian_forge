import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
class SymmetricalLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base, linthresh, linscale):
        super().__init__()
        if base <= 1.0:
            raise ValueError("'base' must be larger than 1")
        if linthresh <= 0.0:
            raise ValueError("'linthresh' must be positive")
        if linscale <= 0.0:
            raise ValueError("'linscale' must be positive")
        self.base = base
        self.linthresh = linthresh
        self.linscale = linscale
        self._linscale_adj = linscale / (1.0 - self.base ** (-1))
        self._log_base = np.log(base)

    @_api.rename_parameter('3.8', 'a', 'values')
    def transform_non_affine(self, values):
        abs_a = np.abs(values)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.sign(values) * self.linthresh * (self._linscale_adj + np.log(abs_a / self.linthresh) / self._log_base)
            inside = abs_a <= self.linthresh
        out[inside] = values[inside] * self._linscale_adj
        return out

    def inverted(self):
        return InvertedSymmetricalLogTransform(self.base, self.linthresh, self.linscale)