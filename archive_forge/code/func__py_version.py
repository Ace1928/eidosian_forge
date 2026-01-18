import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def _py_version() -> str:
    """Get the XGBoost version from Python version file."""
    VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(VERSION_FILE, encoding='ascii') as f:
        return f.read().strip()