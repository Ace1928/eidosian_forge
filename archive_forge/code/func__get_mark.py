import json
import os
import warnings
import tempfile
from functools import wraps
import numpy as np
import numpy.array_api
import numpy.testing as npt
import pytest
import hypothesis
from scipy._lib._fpumode import get_fpu_mode
from scipy._lib._testutils import FPUModeChangeWarning
from scipy._lib import _pep440
from scipy._lib._array_api import SCIPY_ARRAY_API, SCIPY_DEVICE
def _get_mark(item, name):
    if _pep440.parse(pytest.__version__) >= _pep440.Version('3.6.0'):
        mark = item.get_closest_marker(name)
    else:
        mark = item.get_marker(name)
    return mark