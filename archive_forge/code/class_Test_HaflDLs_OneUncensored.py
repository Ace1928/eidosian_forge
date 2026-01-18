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
class Test_HaflDLs_OneUncensored(CheckROSMixin):
    decimal = 3
    res = numpy.array([1.0, 1.0, 12.0, 15.0])
    cen = numpy.array([True, True, True, False])
    rescol = 'value'
    cencol = 'qual'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([0.5, 0.5, 6.0, 15.0])
    expected_cohn = pandas.DataFrame({'nuncen_above': numpy.array([0.0, 1.0, numpy.nan]), 'nobs_below': numpy.array([2.0, 3.0, numpy.nan]), 'ncen_equal': numpy.array([2.0, 1.0, numpy.nan]), 'prob_exceedance': numpy.array([0.25, 0.25, 0.0])})