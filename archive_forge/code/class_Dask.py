import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
class Dask:
    """Class to toggle Dask states.

    Warnings
    --------
    Dask integration is an experimental feature still in progress. It can already be used
    but it doesn't work with all stats nor diagnostics yet.
    """
    dask_flag = False
    dask_kwargs = None

    @classmethod
    def enable_dask(cls, dask_kwargs=None):
        """To enable Dask.

        Parameters
        ----------
        dask_kwargs : dict
            Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.
        """
        cls.dask_flag = True
        cls.dask_kwargs = dask_kwargs

    @classmethod
    def disable_dask(cls):
        """To disable Dask."""
        cls.dask_flag = False
        cls.dask_kwargs = None