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
def _get_additional_properties_error_message(self, error: jsonschema.exceptions.ValidationError) -> str:
    """Output all existing parameters when an unknown parameter is specified."""
    altair_cls = self._get_altair_class_for_error(error)
    param_dict_keys = inspect.signature(altair_cls).parameters.keys()
    param_names_table = self._format_params_as_table(param_dict_keys)
    parameter_name = error.message.split("('")[-1].split("'")[0]
    message = f"`{altair_cls.__name__}` has no parameter named '{parameter_name}'\n\nExisting parameter names are:\n{param_names_table}\nSee the help for `{altair_cls.__name__}` to read the full description of these parameters"
    return message