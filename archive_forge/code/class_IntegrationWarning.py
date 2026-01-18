import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
class IntegrationWarning(UserWarning):
    """
    Warning on issues during integration.
    """
    pass