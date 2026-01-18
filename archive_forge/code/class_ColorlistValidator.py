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
class ColorlistValidator(BaseValidator):
    """
    "colorlist": {
      "description": "A list of colors. Must be an {array} containing
                      valid colors.",
      "requiredOpts": [],
      "otherOpts": [
        "dflt"
      ]
    }
    """

    def __init__(self, plotly_name, parent_name, **kwargs):
        super(ColorlistValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)

    def description(self):
        return "    The '{plotly_name}' property is a colorlist that may be specified\n    as a tuple, list, one-dimensional numpy array, or pandas Series of valid\n    color strings".format(plotly_name=self.plotly_name)

    def validate_coerce(self, v):
        if v is None:
            pass
        elif is_array(v):
            validated_v = [ColorValidator.perform_validate_coerce(e, allow_number=False) for e in v]
            invalid_els = [el for el, validated_el in zip(v, validated_v) if validated_el is None]
            if invalid_els:
                self.raise_invalid_elements(invalid_els)
            v = to_scalar_or_list(v)
        else:
            self.raise_invalid_val(v)
        return v