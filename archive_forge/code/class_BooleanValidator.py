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
class BooleanValidator(BaseValidator):
    """
    "boolean": {
        "description": "A boolean (true/false) value.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, **kwargs):
        super(BooleanValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)

    def description(self):
        return "    The '{plotly_name}' property must be specified as a bool\n    (either True, or False)".format(plotly_name=self.plotly_name)

    def validate_coerce(self, v):
        if v is None:
            pass
        elif not isinstance(v, bool):
            self.raise_invalid_val(v)
        return v