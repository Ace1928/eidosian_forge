import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
class BarsComparisonTest(ComparisonTestCase):

    def setUp(self):
        """Variations on the constructors in the Elements notebook"""
        key_dims1 = [Dimension('Car occupants')]
        key_dims2 = [Dimension('Cyclists')]
        value_dims1 = ['Count']
        self.bars1 = Bars([('one', 8), ('two', 10), ('three', 16)], kdims=key_dims1, vdims=value_dims1)
        self.bars2 = Bars([('one', 8), ('two', 10), ('three', 17)], kdims=key_dims1, vdims=value_dims1)
        self.bars3 = Bars([('one', 8), ('two', 10), ('three', 16)], kdims=key_dims2, vdims=value_dims1)

    def test_bars_equal_1(self):
        self.assertEqual(self.bars1, self.bars1)

    def test_bars_equal_2(self):
        self.assertEqual(self.bars2, self.bars2)

    def test_bars_equal_3(self):
        self.assertEqual(self.bars3, self.bars3)

    def test_bars_unequal_1(self):
        try:
            self.assertEqual(self.bars1, self.bars2)
        except AssertionError as e:
            if 'not almost equal' not in str(e):
                raise Exception(f'Bars mismatched data error not raised. {e}')

    def test_bars_unequal_keydims(self):
        try:
            self.assertEqual(self.bars1, self.bars3)
        except AssertionError as e:
            if not str(e) == 'Dimension names mismatched: Car occupants != Cyclists':
                raise Exception('Bars key dimension mismatch error not raised.')