import operator as op
from numbers import Number
import pytest
import numpy as np
from numpy.polynomial import (
from numpy.testing import (
from numpy.polynomial.polyutils import RankWarning
@pytest.fixture(params=classes, ids=classids)
def Poly(request):
    return request.param