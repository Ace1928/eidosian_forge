from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
class MyDaskArray(da.Array):
    """Dask Array subclass with validation logic in __dask_optimize__"""

    @classmethod
    def __dask_optimize__(cls, dsk, keys, **kwargs):
        assert type(dsk) is HighLevelGraph
        nonlocal optimized
        optimized = True
        return super().__dask_optimize__(dsk, keys, **kwargs)