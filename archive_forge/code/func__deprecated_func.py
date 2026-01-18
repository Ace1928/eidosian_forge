import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@deprecate_func(deprecated_version='x', removed_version='y', hint='You are on your own.')
def _deprecated_func():
    """Dummy function used in `test_deprecate_func`.

    The decorated function must be outside the test function, otherwise it
    seems that the warning does not point at the calling location.
    """