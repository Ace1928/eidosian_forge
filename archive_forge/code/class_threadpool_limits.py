import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
@_format_docstring(USER_APIS=', '.join((f'"{api}"' for api in _ALL_USER_APIS)), BLAS_LIBS=', '.join(_ALL_BLAS_LIBRARIES), OPENMP_LIBS=', '.join(_ALL_OPENMP_LIBRARIES))
class threadpool_limits(_ThreadpoolLimiter):
    """Change the maximal number of threads that can be used in thread pools.

    This object can be used either as a callable (the construction of this object
    limits the number of threads), as a context manager in a `with` block to
    automatically restore the original state of the controlled libraries when exiting
    the block, or as a decorator through its `wrap` method.

    Set the maximal number of threads that can be used in thread pools used in
    the supported libraries to `limit`. This function works for libraries that
    are already loaded in the interpreter and can be changed dynamically.

    This effect is global and impacts the whole Python process. There is no thread level
    isolation as these libraries do not offer thread-local APIs to configure the number
    of threads to use in nested parallel calls.

    Parameters
    ----------
    limits : int, dict, 'sequential_blas_under_openmp' or None (default=None)
        The maximal number of threads that can be used in thread pools

        - If int, sets the maximum number of threads to `limits` for each
          library selected by `user_api`.

        - If it is a dictionary `{{key: max_threads}}`, this function sets a
          custom maximum number of threads for each `key` which can be either a
          `user_api` or a `prefix` for a specific library.

        - If 'sequential_blas_under_openmp', it will chose the appropriate `limits`
          and `user_api` parameters for the specific use case of sequential BLAS
          calls within an OpenMP parallel region. The `user_api` parameter is
          ignored.

        - If None, this function does not do anything.

    user_api : {USER_APIS} or None (default=None)
        APIs of libraries to limit. Used only if `limits` is an int.

        - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

        - If "openmp", it will only limit OpenMP supported libraries
          ({OPENMP_LIBS}). Note that it can affect the number of threads used
          by the BLAS libraries if they rely on OpenMP.

        - If None, this function will apply to all supported libraries.
    """

    def __init__(self, limits=None, user_api=None):
        super().__init__(ThreadpoolController(), limits=limits, user_api=user_api)

    @classmethod
    def wrap(cls, limits=None, user_api=None):
        return super().wrap(ThreadpoolController(), limits=limits, user_api=user_api)