from __future__ import annotations
import copyreg
import multiprocessing
import multiprocessing.pool
import os
import pickle
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from warnings import warn
import cloudpickle
from dask import config
from dask.local import MultiprocessingPoolExecutor, get_async, reraise
from dask.optimization import cull, fuse
from dask.system import CPU_COUNT
from dask.typing import Key
from dask.utils import ensure_dict
class RemoteException(Exception):
    """Remote Exception

    Contains the exception and traceback from a remotely run task
    """

    def __init__(self, exception, traceback):
        self.exception = exception
        self.traceback = traceback

    def __str__(self):
        return str(self.exception) + '\n\nTraceback\n---------\n' + self.traceback

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__) + dir(self.exception)))

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            return getattr(self.exception, key)