import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
class IntegerValidator(BaseValidator):
    """
    "integer": {
        "description": "An integer or an integer inside a string. When
                        applicable, values greater (less) than `max`
                        (`min`) are coerced to the `dflt`.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt",
            "min",
            "max",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, min=None, max=None, array_ok=False, **kwargs):
        super(IntegerValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        if min is None and max is not None:
            self.min_val = -sys.maxsize - 1
        else:
            self.min_val = min
        if max is None and min is not None:
            self.max_val = sys.maxsize
        else:
            self.max_val = max
        if min is not None or max is not None:
            self.has_min_max = True
        else:
            self.has_min_max = False
        self.array_ok = array_ok

    def description(self):
        desc = "    The '{plotly_name}' property is a integer and may be specified as:".format(plotly_name=self.plotly_name)
        if not self.has_min_max:
            desc = desc + '\n      - An int (or float that will be cast to an int)'
        else:
            desc = desc + '\n      - An int (or float that will be cast to an int)\n        in the interval [{min_val}, {max_val}]'.format(min_val=self.min_val, max_val=self.max_val)
        if self.array_ok:
            desc = desc + '\n      - A tuple, list, or one-dimensional numpy array of the above'
        return desc

    def validate_coerce(self, v):
        if v is None:
            pass
        elif self.array_ok and is_homogeneous_array(v):
            np = get_module('numpy')
            v_array = copy_to_readonly_numpy_array(v, kind=('i', 'u'), force_numeric=True)
            if v_array.dtype.kind not in ['i', 'u']:
                self.raise_invalid_val(v)
            if self.has_min_max:
                v_valid = np.logical_and(self.min_val <= v_array, v_array <= self.max_val)
                if not np.all(v_valid):
                    v_invalid = np.logical_not(v_valid)
                    some_invalid_els = np.array(v, dtype='object')[v_invalid][:10].tolist()
                    self.raise_invalid_elements(some_invalid_els)
            v = v_array
        elif self.array_ok and is_simple_array(v):
            invalid_els = [e for e in v if not isinstance(e, int)]
            if invalid_els:
                self.raise_invalid_elements(invalid_els[:10])
            if self.has_min_max:
                invalid_els = [e for e in v if not self.min_val <= e <= self.max_val]
                if invalid_els:
                    self.raise_invalid_elements(invalid_els[:10])
            v = to_scalar_or_list(v)
        else:
            if not isinstance(v, int):
                self.raise_invalid_val(v)
            if self.has_min_max:
                if not self.min_val <= v <= self.max_val:
                    self.raise_invalid_val(v)
        return v