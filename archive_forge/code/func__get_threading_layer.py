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
def _get_threading_layer(self):
    """Return the threading layer of MKL"""
    set_threading_layer = getattr(self.dynlib, 'MKL_Set_Threading_Layer', lambda layer: -1)
    layer_map = {0: 'intel', 1: 'sequential', 2: 'pgi', 3: 'gnu', 4: 'tbb', -1: 'not specified'}
    return layer_map[set_threading_layer(-1)]