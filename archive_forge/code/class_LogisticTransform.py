import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
class LogisticTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, nonpositive='mask'):
        super().__init__()
        self._nonpositive = nonpositive

    @_api.rename_parameter('3.8', 'a', 'values')
    def transform_non_affine(self, values):
        """logistic transform (base 10)"""
        return 1.0 / (1 + 10 ** (-values))

    def inverted(self):
        return LogitTransform(self._nonpositive)

    def __str__(self):
        return f'{type(self).__name__}({self._nonpositive!r})'