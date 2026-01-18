import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def _make_axis_spanning_layout_object(self, direction, shape):
    """
        Convert a shape drawn on a plot or a subplot into one whose yref or xref
        ends with " domain" and has coordinates so that the shape will seem to
        extend infinitely in that dimension. This is useful for drawing lines or
        boxes on a plot where one dimension of the shape will not move out of
        bounds when moving the plot's view.
        Note that the shape already added to the (sub)plot must have the
        corresponding axis reference referring to an actual axis (e.g., 'x',
        'y2' etc. are accepted, but not 'paper'). This will be the case if the
        shape was added with "add_shape".
        Shape must have the x0, x1, y0, y1 fields already initialized.
        """
    if direction == 'vertical':
        ref = 'yref'
    elif direction == 'horizontal':
        ref = 'xref'
    else:
        raise ValueError("Bad direction: %s. Permissible values are 'vertical' and 'horizontal'." % (direction,))
    shape[ref] += ' domain'
    return shape