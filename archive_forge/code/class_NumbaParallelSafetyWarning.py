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
class NumbaParallelSafetyWarning(NumbaWarning):
    """
    Warning category for when an operation in a prange
    might not have parallel semantics.
    """