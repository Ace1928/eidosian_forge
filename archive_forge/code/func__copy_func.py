import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def _copy_func(f):
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g