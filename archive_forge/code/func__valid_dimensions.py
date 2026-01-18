import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
def _valid_dimensions(self, dimensions):
    """Validates key dimension input

        Returns kdims if no dimensions are specified"""
    if dimensions is None:
        dimensions = self.kdims
    elif not isinstance(dimensions, list):
        dimensions = [dimensions]
    valid_dimensions = []
    for dim in dimensions:
        if isinstance(dim, Dimension):
            dim = dim.name
        if dim not in self.kdims:
            raise Exception(f'Supplied dimensions {dim} not found.')
        valid_dimensions.append(dim)
    return valid_dimensions