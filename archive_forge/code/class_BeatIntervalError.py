from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
class BeatIntervalError(Exception):
    """
    Exception to be raised whenever an interval cannot be computed.

    """

    def __init__(self, value=None):
        if value is None:
            value = 'At least two beats must be present to be able to calculate an interval.'
        self.value = value

    def __str__(self):
        return repr(self.value)