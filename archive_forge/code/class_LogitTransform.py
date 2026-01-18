import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
class LogitTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, nonpositive='mask'):
        super().__init__()
        _api.check_in_list(['mask', 'clip'], nonpositive=nonpositive)
        self._nonpositive = nonpositive
        self._clip = {'clip': True, 'mask': False}[nonpositive]

    @_api.rename_parameter('3.8', 'a', 'values')
    def transform_non_affine(self, values):
        """logit transform (base 10), masked or clipped"""
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.log10(values / (1 - values))
        if self._clip:
            out[values <= 0] = -1000
            out[1 <= values] = 1000
        return out

    def inverted(self):
        return LogisticTransform(self._nonpositive)

    def __str__(self):
        return f'{type(self).__name__}({self._nonpositive!r})'