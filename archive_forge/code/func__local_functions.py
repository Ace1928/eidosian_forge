from __future__ import annotations
import dataclasses
import datetime
import decimal
import operator
import pathlib
import pickle
import random
import subprocess
import sys
import textwrap
from enum import Enum, Flag, IntEnum, IntFlag
from typing import Union
import cloudpickle
import pytest
from tlz import compose, curry, partial
import dask
from dask.base import TokenizationError, normalize_token, tokenize
from dask.core import literal
from dask.utils import tmpfile
from dask.utils_test import import_or_none
def _local_functions():
    all_funcs = [lambda x: x, lambda x: x + 1, lambda y: y, lambda y: y + 1]
    a, b = all_funcs[:2]

    def func(x):
        return x

    def f2(x):
        return x
    all_funcs += [func, f2]
    local_scope = 1

    def func():
        nonlocal local_scope
        local_scope += 1
        return a(local_scope)
    all_funcs.append(func)

    def func():
        global _GLOBAL
        _GLOBAL += 1
        return _GLOBAL
    all_funcs.append(func)

    def func(x, c=a):
        return c(x)
    all_funcs.append(func)

    def func(x, c=b):
        return c(x)
    all_funcs.append(func)

    def func(x):
        c = lambda x: x + 2
        return c(x)
    all_funcs.append(func)

    def func(x):
        c = lambda x: x + 3
        return c(x)
    all_funcs.append(func)

    def func(x):
        c = a
        return c(x)
    all_funcs.append(func)

    def func(x):
        c = b
        return c(x)
    all_funcs.append(func)
    return all_funcs