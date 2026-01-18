import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@deprecate_parameter('old1', start_version='0.10', stop_version='0.12')
@deprecate_parameter('old0', start_version='0.10', stop_version='0.12')
def _func_deprecated_params(arg0, old0=DEPRECATED, old1=DEPRECATED, arg1=None):
    """Expected docstring.

    Parameters
    ----------
    arg0 : int
        First unchanged parameter.
    arg1 : int, optional
        Second unchanged parameter.
    """
    return (arg0, old0, old1, arg1)