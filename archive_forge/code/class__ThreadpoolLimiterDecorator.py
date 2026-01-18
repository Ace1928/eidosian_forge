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
class _ThreadpoolLimiterDecorator(_ThreadpoolLimiter, ContextDecorator):
    """Same as _ThreadpoolLimiter but to be used as a decorator"""

    def __init__(self, controller, *, limits=None, user_api=None):
        self._limits, self._user_api, self._prefixes = self._check_params(limits, user_api)
        self._controller = controller

    def __enter__(self):
        self._original_info = self._controller.info()
        self._set_threadpool_limits()
        return self