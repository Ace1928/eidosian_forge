from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
def _resolve_doubly_passed_kwarg(kwargs_dict: dict[Any, Any], kwarg_name: str, passed_kwarg_value: str | bool | None, default: bool | None, err_msg_dict_name: str) -> dict[Any, Any]:
    if kwarg_name in kwargs_dict and passed_kwarg_value is None:
        pass
    elif kwarg_name not in kwargs_dict and passed_kwarg_value is not None:
        kwargs_dict[kwarg_name] = passed_kwarg_value
    elif kwarg_name not in kwargs_dict and passed_kwarg_value is None:
        kwargs_dict[kwarg_name] = default
    else:
        raise ValueError(f'argument {kwarg_name} cannot be passed both as a keyword argument and within the {err_msg_dict_name} dictionary')
    return kwargs_dict