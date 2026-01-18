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
def _set_array_prop(self, prop, val):
    """
        Set the value of a compound property

        Parameters
        ----------
        prop : str
            Name of a compound property
        val
            The new property value

        Returns
        -------
        tuple[BasePlotlyType]
            The coerced assigned object
        """
    if val is Undefined:
        return
    validator = self._get_validator(prop)
    val = validator.validate_coerce(val, skip_invalid=self._skip_invalid)
    curr_val = self._compound_array_props.get(prop, None)
    if curr_val is not None:
        curr_dict_vals = [deepcopy(cv._props) for cv in curr_val]
    else:
        curr_dict_vals = None
    if val is not None:
        new_dict_vals = [deepcopy(nv._props) for nv in val]
    else:
        new_dict_vals = None
    if not self._in_batch_mode:
        if not new_dict_vals:
            if self._props and prop in self._props:
                self._props.pop(prop)
        else:
            self._init_props()
            self._props[prop] = new_dict_vals
    if not BasePlotlyType._vals_equal(curr_dict_vals, new_dict_vals):
        self._send_prop_set(prop, new_dict_vals)
    if val is not None:
        for v in val:
            v._orphan_props.clear()
            v._parent = self
    if curr_val is not None:
        for cv, cv_dict in zip(curr_val, curr_dict_vals):
            if cv_dict is not None:
                cv._orphan_props.update(cv_dict)
            cv._parent = None
    self._compound_array_props[prop] = val
    return val