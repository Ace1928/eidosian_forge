import collections
import os
import pickle
from abc import ABC
from typing import (
import numpy
from . import collective
from .core import Booster, DMatrix, XGBoostError, _parse_eval_str
def _allreduce_metric(score: _ART) -> _ART:
    """Helper function for computing customized metric in distributed
    environment.  Not strictly correct as many functions don't use mean value
    as final result.

    """
    world = collective.get_world_size()
    assert world != 0
    if world == 1:
        return score
    if isinstance(score, tuple):
        raise ValueError('xgboost.cv function should not be used in distributed environment.')
    arr = numpy.array([score])
    arr = collective.allreduce(arr, collective.Op.SUM) / world
    return arr[0]