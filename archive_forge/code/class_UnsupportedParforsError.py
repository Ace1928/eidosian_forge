import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class UnsupportedParforsError(NumbaError):
    """
    An error occurred because parfors is not supported on the platform.
    """
    pass