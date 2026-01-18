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
class Test_ROS_HelselArsenic(CheckROSMixin):
    """
    Oahu arsenic data from Nondetects and Data Analysis by
    Dennis R. Helsel (John Wiley, 2005)

    Plotting positions are fudged since relative to source data since
    modeled data is what matters and (source data plot positions are
    not uniformly spaced, which seems weird)
    """
    decimal = 2
    res = numpy.array([3.2, 2.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.7, 1.5, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5])
    cen = numpy.array([False, False, True, True, True, True, True, True, True, True, False, False, True, True, True, True, False, True, False, False, False, False, False, False])
    rescol = 'obs'
    cencol = 'cen'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([3.2, 2.8, 1.42, 1.14, 0.95, 0.81, 0.68, 0.57, 0.46, 0.35, 1.7, 1.5, 0.98, 0.76, 0.58, 0.41, 0.9, 0.61, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5])
    expected_cohn = pandas.DataFrame({'nuncen_above': numpy.array([6.0, 1.0, 2.0, 2.0, numpy.nan]), 'nobs_below': numpy.array([0.0, 7.0, 12.0, 22.0, numpy.nan]), 'ncen_equal': numpy.array([0.0, 1.0, 4.0, 8.0, numpy.nan]), 'prob_exceedance': numpy.array([1.0, 0.3125, 0.21429, 0.0833, 0.0])})