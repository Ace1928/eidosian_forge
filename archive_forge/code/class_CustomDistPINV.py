import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
class CustomDistPINV:

    def __init__(self, pdf, args):
        self._pdf = lambda x: pdf(x, *args)

    def pdf(self, x):
        return self._pdf(x)