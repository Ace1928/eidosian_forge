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
def _json_path(err: jsonschema.exceptions.ValidationError) -> str:
    """Drop in replacement for the .json_path property of the jsonschema
    ValidationError class, which is not available as property for
    ValidationError with jsonschema<4.0.1.
    More info, see https://github.com/altair-viz/altair/issues/3038
    """
    path = '$'
    for elem in err.absolute_path:
        if isinstance(elem, int):
            path += '[' + str(elem) + ']'
        else:
            path += '.' + elem
    return path