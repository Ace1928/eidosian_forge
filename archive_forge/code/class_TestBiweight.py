import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest
from statsmodels.sandbox.nonparametric import kernels
class TestBiweight(CheckKernelMixin):
    kern_name = 'bi'
    kern = kernels.Biweight()
    se_n_diff = 9
    low_rtol = 0.3