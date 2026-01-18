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
def _init_child_props(self, child):
    """
        Ensure that a properties dict has been initialized for a child object

        Parameters
        ----------
        child : BasePlotlyType

        Returns
        -------
        None
        """
    self._init_props()
    if child.plotly_name in self._compound_props:
        if child.plotly_name not in self._props:
            self._props[child.plotly_name] = {}
    elif child.plotly_name in self._compound_array_props:
        children = self._compound_array_props[child.plotly_name]
        child_ind = BaseFigure._index_is(children, child)
        assert child_ind is not None
        if child.plotly_name not in self._props:
            self._props[child.plotly_name] = []
        children_list = self._props[child.plotly_name]
        while len(children_list) <= child_ind:
            children_list.append({})
    else:
        raise ValueError('Invalid child with name: %s' % child.plotly_name)