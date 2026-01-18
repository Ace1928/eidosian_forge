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
class RedefinedError(IRError):
    """
    An error occurred during interpretation of IR due to variable redefinition.
    """
    pass