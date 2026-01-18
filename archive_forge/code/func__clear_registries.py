import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
def _clear_registries(self):
    if hasattr(warnings, '_filters_mutated'):
        warnings._filters_mutated()
        return
    for module in self._tmp_modules:
        if hasattr(module, '__warningregistry__'):
            module.__warningregistry__.clear()