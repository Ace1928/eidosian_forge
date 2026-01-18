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
def _subplot_not_empty(self, xref, yref, selector='all'):
    """
        xref: string representing the axis. Objects in the plot will be checked
              for this xref (for layout objects) or xaxis (for traces) to
              determine if they lie in a certain subplot.
        yref: string representing the axis. Objects in the plot will be checked
              for this yref (for layout objects) or yaxis (for traces) to
              determine if they lie in a certain subplot.
        selector: can be "all" or an iterable containing some combination of
                  "traces", "shapes", "annotations", "images". Only the presence
                  of objects specified in selector will be checked. So if
                  ["traces","shapes"] is passed then a plot we be considered
                  non-empty if it contains traces or shapes. If
                  bool(selector) returns False, no checking is performed and
                  this function returns True. If selector is True, it is
                  converted to "all".
        """
    if not selector:
        return True
    if selector is True:
        selector = 'all'
    if selector == 'all':
        selector = ['traces', 'shapes', 'annotations', 'images']
    ret = False
    for s in selector:
        if s == 'traces':
            obj = self.data
            xaxiskw = 'xaxis'
            yaxiskw = 'yaxis'
        elif s in ['shapes', 'annotations', 'images']:
            obj = self.layout[s]
            xaxiskw = 'xref'
            yaxiskw = 'yref'
        else:
            obj = None
        if obj:
            ret |= any((t == (xref, yref) for t in [('x' if d[xaxiskw] is None else d[xaxiskw], 'y' if d[yaxiskw] is None else d[yaxiskw]) for d in obj]))
    return ret