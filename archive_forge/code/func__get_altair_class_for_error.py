import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
def _get_altair_class_for_error(self, error: jsonschema.exceptions.ValidationError) -> Type['SchemaBase']:
    """Try to get the lowest class possible in the chart hierarchy so
        it can be displayed in the error message. This should lead to more informative
        error messages pointing the user closer to the source of the issue.
        """
    for prop_name in reversed(error.absolute_path):
        if isinstance(prop_name, str):
            potential_class_name = prop_name[0].upper() + prop_name[1:]
            cls = getattr(vegalite, potential_class_name, None)
            if cls is not None:
                break
    else:
        cls = self.obj.__class__
    return cls