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
def _indexing_combinations(dims, alls, product=False):
    """
    Gives indexing tuples specified by the coordinates in dims.
    If a member of dims is 'all' then it is replaced by the corresponding member
    in alls.
    If product is True, then the cartesian product of all the indices is
    returned, otherwise the zip (that means index lists of mis-matched length
    will yield a list of tuples whose length is the length of the shortest
    list).
    """
    if len(dims) == 0:
        return []
    if len(dims) != len(alls):
        raise ValueError('Must have corresponding values in alls for each value of dims. Got dims=%s and alls=%s.' % (str(dims), str(alls)))
    r = []
    for d, a in zip(dims, alls):
        if d == 'all':
            d = a
        elif not isinstance(d, list):
            d = [d]
        r.append(d)
    if product:
        return itertools.product(*r)
    else:
        return zip(*r)