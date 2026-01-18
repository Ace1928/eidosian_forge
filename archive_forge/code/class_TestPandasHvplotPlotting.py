from unittest import TestCase, SkipTest
import holoviews as hv
import pandas as pd
import pytest
from packaging.version import Version
from parameterized import parameterized
from hvplot.converter import HoloViewsConverter
from hvplot.plotting import plot
from hvplot.tests.util import makeDataFrame
class TestPandasHvplotPlotting(TestPandasHoloviewsPlotting):

    def setUp(self):
        if Version(pd.__version__) < Version('0.25.1'):
            raise SkipTest('entrypoints for plotting.backends was added in pandas 0.25.1')
        pd.options.plotting.backend = 'hvplot'