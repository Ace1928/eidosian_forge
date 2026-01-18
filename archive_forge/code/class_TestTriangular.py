import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest
from statsmodels.sandbox.nonparametric import kernels
class TestTriangular(CheckKernelMixin):
    kern_name = 'tri'
    kern = kernels.Triangular()
    se_n_diff = 10
    upp_rtol = 0.15
    low_rtol = 0.3