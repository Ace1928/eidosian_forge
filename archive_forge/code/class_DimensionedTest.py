import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
class DimensionedTest(ComparisonTestCase):

    def test_dimensioned_init(self):
        Dimensioned('An example of arbitrary data')

    def test_dimensioned_constant_label(self):
        label = 'label'
        view = Dimensioned('An example of arbitrary data', label=label)
        self.assertEqual(view.label, label)
        try:
            view.label = 'another label'
            raise AssertionError('Label should be a constant parameter.')
        except TypeError:
            pass

    def test_dimensioned_redim_string(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x='Test'))

    def test_dimensioned_redim_dict_label(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=[('x', 'Test')])
        self.assertEqual(redimensioned, dimensioned.redim.label(x='Test'))

    def test_dimensioned_redim_dict_label_existing_error(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=[('x', 'Test1')])
        with self.assertRaisesRegex(ValueError, 'Cannot override an existing Dimension label'):
            dimensioned.redim.label(x='Test2')

    def test_dimensioned_redim_dimension(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x=Dimension('Test')))

    def test_dimensioned_redim_dict(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x={'name': 'Test'}))

    def test_dimensioned_redim_dict_range(self):
        redimensioned = Dimensioned('Arbitrary Data', kdims=['x']).redim(x={'range': (0, 10)})
        self.assertEqual(redimensioned.kdims[0].range, (0, 10))

    def test_dimensioned_redim_range_aux(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.redim.range(x=(-10, 42))
        self.assertEqual(redimensioned.kdims[0].range, (-10, 42))

    def test_dimensioned_redim_cyclic_aux(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.redim.cyclic(x=True)
        self.assertEqual(redimensioned.kdims[0].cyclic, True)