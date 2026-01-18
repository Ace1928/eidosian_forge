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
class AnyValidator(BaseValidator):
    """
    "any": {
        "description": "Any type.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt",
            "values",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, values=None, array_ok=False, **kwargs):
        super(AnyValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.values = values
        self.array_ok = array_ok

    def description(self):
        desc = "    The '{plotly_name}' property accepts values of any type\n        ".format(plotly_name=self.plotly_name)
        return desc

    def validate_coerce(self, v):
        if v is None:
            pass
        elif self.array_ok and is_homogeneous_array(v):
            v = copy_to_readonly_numpy_array(v, kind='O')
        elif self.array_ok and is_simple_array(v):
            v = to_scalar_or_list(v)
        return v