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
def _set_subplotid_prop(self, prop, value):
    """
        Set a subplot property on the layout

        Parameters
        ----------
        prop : str
            A valid subplot property
        value
            Subplot value
        """
    match = self._subplot_re_match(prop)
    subplot_prop = match.group(1)
    suffix_digit = int(match.group(2))
    if suffix_digit == 0:
        raise TypeError('Subplot properties may only be suffixed by an integer >= 1\nReceived {k}'.format(k=prop))
    if suffix_digit == 1:
        prop = subplot_prop
    if prop not in self._valid_props:
        self._valid_props.add(prop)
    self._set_compound_prop(prop, value)
    self._subplotid_props.add(prop)