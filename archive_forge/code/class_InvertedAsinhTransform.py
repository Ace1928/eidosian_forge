import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
class InvertedAsinhTransform(Transform):
    """Hyperbolic sine transformation used by `.AsinhScale`"""
    input_dims = output_dims = 1

    def __init__(self, linear_width):
        super().__init__()
        self.linear_width = linear_width

    @_api.rename_parameter('3.8', 'a', 'values')
    def transform_non_affine(self, values):
        return self.linear_width * np.sinh(values / self.linear_width)

    def inverted(self):
        return AsinhTransform(self.linear_width)