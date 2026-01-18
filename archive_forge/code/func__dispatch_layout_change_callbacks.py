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
def _dispatch_layout_change_callbacks(self, relayout_data):
    """
        Dispatch property change callbacks given relayout_data

        Parameters
        ----------
        relayout_data : dict[str, any]
            See docstring for plotly_relayout.

        Returns
        -------
        None
        """
    key_path_strs = list(relayout_data.keys())
    dispatch_plan = BaseFigure._build_dispatch_plan(key_path_strs)
    for path_tuple, changed_paths in dispatch_plan.items():
        if path_tuple in self.layout:
            dispatch_obj = self.layout[path_tuple]
            if isinstance(dispatch_obj, BasePlotlyType):
                dispatch_obj._dispatch_change_callbacks(changed_paths)