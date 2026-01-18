import plotly.graph_objs as go
from _plotly_utils.basevalidators import ColorscaleValidator
from ._core import apply_default_cascade, init_figure, configure_animation_controls
from .imshow_utils import rescale_intensity, _integer_ranges, _integer_types
import pandas as pd
import numpy as np
import itertools
from plotly.utils import image_array_to_data_uri
def _vectorize_zvalue(z, mode='max'):
    alpha = 255 if mode == 'max' else 0
    if z is None:
        return z
    elif np.isscalar(z):
        return [z] * 3 + [alpha]
    elif len(z) == 1:
        return list(z) * 3 + [alpha]
    elif len(z) == 3:
        return list(z) + [alpha]
    elif len(z) == 4:
        return z
    else:
        raise ValueError('zmax can be a scalar, or an iterable of length 1, 3 or 4. A value of %s was passed for zmax.' % str(z))