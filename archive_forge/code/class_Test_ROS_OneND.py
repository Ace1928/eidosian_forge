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
class Test_ROS_OneND(CheckROSMixin):
    decimal = 3
    res = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 3.0, 7.0, 9.0, 12.0, 15.0, 20.0, 27.0, 33.0, 50.0])
    cen = numpy.array([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    rescol = 'conc'
    cencol = 'cen'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([0.24, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 3.0, 7.0, 9.0, 12.0, 15.0, 20.0, 27.0, 33.0, 50.0])
    expected_cohn = pandas.DataFrame({'nuncen_above': numpy.array([17.0, numpy.nan]), 'nobs_below': numpy.array([1.0, numpy.nan]), 'ncen_equal': numpy.array([1.0, numpy.nan]), 'prob_exceedance': numpy.array([0.94444, 0.0])})