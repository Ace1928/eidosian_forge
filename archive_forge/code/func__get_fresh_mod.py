import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def _get_fresh_mod():
    my_mod = sys.modules[__name__]
    try:
        my_mod.__warningregistry__.clear()
    except AttributeError:
        pass
    return my_mod