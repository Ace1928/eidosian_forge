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
def _get_leaves_of_error_tree(errors: ValidationErrorList) -> ValidationErrorList:
    """For each error in `errors`, it traverses down the "error tree" that is generated
    by the jsonschema library to find and return all "leaf" errors. These are errors
    which have no further errors that caused it and so they are the most specific errors
    with the most specific error messages.
    """
    leaves: ValidationErrorList = []
    for err in errors:
        if err.context:
            leaves.extend(_get_leaves_of_error_tree(err.context))
        else:
            leaves.append(err)
    return leaves