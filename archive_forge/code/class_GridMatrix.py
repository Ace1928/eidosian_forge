import itertools
import types
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import param
from ..streams import Params, Stream, streams_list_from_dict
from . import traversal, util
from .accessors import Opts, Redim
from .dimension import Dimension, ViewableElement
from .layout import AdjointLayout, Empty, Layout, Layoutable, NdLayout
from .ndmapping import NdMapping, UniformNdMapping, item_check
from .options import Store, StoreOptions
from .overlay import CompositeOverlay, NdOverlay, Overlay, Overlayable
class GridMatrix(GridSpace):
    """
    GridMatrix is container type for heterogeneous Element types
    laid out in a grid. Unlike a GridSpace the axes of the Grid
    must not represent an actual coordinate space, but may be used
    to plot various dimensions against each other. The GridMatrix
    is usually constructed using the gridmatrix operation, which
    will generate a GridMatrix plotting each dimension in an
    Element against each other.
    """

    def _item_check(self, dim_vals, data):
        if not traversal.uniform(NdMapping([(0, self), (1, data)])):
            raise ValueError('HoloMaps dimensions must be consistent in %s.' % type(self).__name__)
        NdMapping._item_check(self, dim_vals, data)