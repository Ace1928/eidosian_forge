from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal
from io import StringIO
from textwrap import dedent
import numpy as np
import numpy.testing as npt
import numpy
from numpy.testing import assert_equal
import pandas
import pytest
from statsmodels.imputation import ros
class Test_ROS_OnlyDL_GT_MaxUncen(Test_NoOp_ZeroND):
    numpy.random.seed(0)
    N = 20
    res = [0.38, 0.43, 0.81, 0.86, 0.9, 1.13, 1.15, 1.37, 1.4, 1.49, 1.51, 1.56, 2.14, 2.59, 2.66, 4.28, 4.46, 5.84, 6.47, 9.4, 10.0, 10.0]
    cen = [False] * N + [True, True]