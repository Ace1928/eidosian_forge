import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
class DimensionDefaultTest(ComparisonTestCase):

    def test_validate_default_against_values(self):
        msg = "Dimension\\('A'\\) default 1\\.1 not found in declared values: \\[0, 1\\]"
        with self.assertRaisesRegex(ValueError, msg):
            Dimension('A', values=[0, 1], default=1.1)

    def test_validate_default_against_range(self):
        msg = "Dimension\\('A'\\) default 1\\.1 not in declared range: \\(0, 1\\)"
        with self.assertRaisesRegex(ValueError, msg):
            Dimension('A', range=(0, 1), default=1.1)