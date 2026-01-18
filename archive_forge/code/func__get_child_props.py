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
def _get_child_props(self, child):
    """
        Return the properties dict for a child trace or child layout

        Note: this method must match the name/signature of one on
        BasePlotlyType

        Parameters
        ----------
        child : BaseTraceType | BaseLayoutType

        Returns
        -------
        dict
        """
    try:
        trace_index = BaseFigure._index_is(self.data, child)
    except ValueError:
        trace_index = None
    if trace_index is not None:
        if 'data' in self._props:
            return self._props['data'][trace_index]
        else:
            return None
    elif child is self.layout:
        return self._props.get('layout', None)
    else:
        raise ValueError('Unrecognized child: %s' % child)