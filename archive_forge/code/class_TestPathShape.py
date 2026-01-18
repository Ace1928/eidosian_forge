import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
class TestPathShape(TestShape):

    def assert_path_shape_element(self, shape, element, xref='x', yref='y'):
        self.assertEqual(shape['type'], 'path')
        expected_path = 'M' + 'L'.join([f'{x} {y}' for x, y in zip(element.dimension_values(0), element.dimension_values(1))]) + 'Z'
        self.assertEqual(shape['path'], expected_path)
        self.assertEqual(shape['xref'], xref)
        self.assertEqual(shape['yref'], yref)

    def test_simple_path(self):
        path = Path([(0, 0), (1, 1), (0, 1), (0, 0)])
        state = self._get_plot_state(path)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 1)
        self.assert_path_shape_element(shapes[0], path)
        self.assert_shape_element_styling(path)