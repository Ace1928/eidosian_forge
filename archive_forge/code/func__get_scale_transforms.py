from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
def _get_scale_transforms(self, axis):
    """Return a function implementing the scale transform (or its inverse)."""
    if self.ax is None:
        axis_list = [getattr(ax, f'{axis}axis') for ax in self.facets.axes.flat]
        scales = {axis.get_scale() for axis in axis_list}
        if len(scales) > 1:
            err = 'Cannot determine transform with mixed scales on faceted axes.'
            raise RuntimeError(err)
        transform_obj = axis_list[0].get_transform()
    else:
        transform_obj = getattr(self.ax, f'{axis}axis').get_transform()
    return (transform_obj.transform, transform_obj.inverted().transform)