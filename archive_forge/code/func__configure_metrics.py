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
def _configure_metrics(params: BoosterParam) -> BoosterParam:
    if isinstance(params, dict) and 'eval_metric' in params and isinstance(params['eval_metric'], list):
        eval_metrics = params['eval_metric']
        params.pop('eval_metric', None)
        params_list = list(params.items())
        for eval_metric in eval_metrics:
            params_list += [('eval_metric', eval_metric)]
        return params_list
    return params