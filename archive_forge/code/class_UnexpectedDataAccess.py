from __future__ import annotations
import importlib
import platform
import string
import warnings
from contextlib import contextmanager, nullcontext
from unittest import mock  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal  # noqa: F401
from packaging.version import Version
from pandas.testing import assert_frame_equal  # noqa: F401
import xarray.testing
from xarray import Dataset
from xarray.core import utils
from xarray.core.duck_array_ops import allclose_or_equiv  # noqa: F401
from xarray.core.indexing import ExplicitlyIndexed
from xarray.core.options import set_options
from xarray.core.variable import IndexVariable
from xarray.testing import (  # noqa: F401
class UnexpectedDataAccess(Exception):
    pass