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
def _integer_repr(x, vdt, comp):
    rx = x.view(vdt)
    if not rx.size == 1:
        rx[rx < 0] = comp - rx[rx < 0]
    elif rx < 0:
        rx = comp - rx
    return rx