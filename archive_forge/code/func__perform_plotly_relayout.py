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
def _perform_plotly_relayout(self, relayout_data):
    """
        Perform a relayout operation on the figure's layout data and return
        the changes that were applied

        Parameters
        ----------
        relayout_data : dict[str, any]
            See the docstring for plotly_relayout
        Returns
        -------
        relayout_changes: dict[str, any]
            Subset of relayout_data including only the keys / values that
            resulted in a change to the figure's layout data
        """
    relayout_changes = {}
    for key_path_str, v in relayout_data.items():
        if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):
            raise ValueError("\nInvalid property path '{key_path_str}' for layout\n".format(key_path_str=key_path_str))
        val_changed = BaseFigure._set_in(self._layout, key_path_str, v)
        if val_changed:
            relayout_changes[key_path_str] = v
    return relayout_changes