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
class LiteralValidator(BaseValidator):
    """
    Validator for readonly literal values
    """

    def __init__(self, plotly_name, parent_name, val, **kwargs):
        super(LiteralValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.val = val

    def validate_coerce(self, v):
        if v != self.val:
            raise ValueError("    The '{plotly_name}' property of {parent_name} is read-only".format(plotly_name=self.plotly_name, parent_name=self.parent_name))
        else:
            return v