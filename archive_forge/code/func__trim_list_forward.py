import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def _trim_list_forward(model, Xs, is_train):

    def backprop(dYs):
        dXs = []
        for dY in dYs:
            dXs.append(model.ops.alloc2f(dY.shape[0], dY.shape[1] + 1))
        return dXs
    Ys = [X[:, :-1] for X in Xs]
    return (Ys, backprop)