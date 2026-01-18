from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def _get_registered_option(key: str):
    """
    Retrieves the option metadata if `key` is a registered option.

    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """
    return _registered_options.get(key)