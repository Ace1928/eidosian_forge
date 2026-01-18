from typing import List, Optional
import numpy
import pytest
import srsly
from numpy.testing import assert_almost_equal
from thinc.api import Dropout, Model, NumpyOps, registry, with_padded
from thinc.backends import NumpyOps
from thinc.compat import has_torch
from thinc.types import Array2d, Floats2d, FloatsXd, Padded, Ragged, Shape
from thinc.util import data_validation, get_width
class NoDropoutOps(NumpyOps):

    def get_dropout_mask(self, shape: Shape, drop: Optional[float]) -> FloatsXd:
        if drop is None or drop <= 0:
            return self.xp.ones(shape, dtype='f')
        else:
            raise ValueError('During prediction, dropout should not be applied')