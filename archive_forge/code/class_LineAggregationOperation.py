import warnings
from collections.abc import Callable, Iterable
from functools import partial
import dask.dataframe as dd
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import param
import xarray as xr
from datashader.colors import color_lookup
from packaging.version import Version
from param.parameterized import bothmethod
from ..core import (
from ..core.data import (
from ..core.util import (
from ..element import (
from ..element.util import connect_tri_edges_pd
from ..streams import PointerXY
from .resample import LinkableOperation, ResampleOperation2D
class LineAggregationOperation(AggregationOperation):
    line_width = param.Number(default=None, bounds=(0, None), doc='\n        Width of the line to draw, in pixels. If zero, the default,\n        lines are drawn using a simple algorithm with a blocky\n        single-pixel width based on whether the line passes through\n        each pixel or does not. If greater than one, lines are drawn\n        with the specified width using a slower and more complex\n        antialiasing algorithm with fractional values along each edge,\n        so that lines have a more uniform visual appearance across all\n        angles. Line widths between 0 and 1 effectively use a\n        line_width of 1 pixel but with a proportionate reduction in\n        the strength of each pixel, approximating the visual\n        appearance of a subpixel line width.')