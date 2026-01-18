import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
class ElementConstructorTest(ComparisonTestCase):
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def setUp(self):
        self.xs = np.linspace(0, 2 * np.pi, 11)
        self.hxs = np.arange(len(self.xs))
        self.sin = np.sin(self.xs)
        self.cos = np.cos(self.xs)
        sine_data = np.column_stack((self.xs, self.sin))
        cos_data = np.column_stack((self.xs, self.cos))
        self.curve = Curve(sine_data)
        self.path = Path([sine_data, cos_data])
        self.histogram = Histogram((self.hxs, self.sin))
        super().setUp()

    def test_empty_element_constructor(self):
        failed_elements = []
        for name, el in param.concrete_descendents(Element).items():
            if name == 'Sankey':
                continue
            if issubclass(el, (Annotation, BaseShape, Div, Tiles)):
                continue
            try:
                el([])
            except Exception:
                failed_elements.append(name)
        self.assertEqual(failed_elements, [])

    def test_chart_zipconstruct(self):
        self.assertEqual(Curve(zip(self.xs, self.sin)), self.curve)

    def test_chart_tuple_construct(self):
        self.assertEqual(Curve((self.xs, self.sin)), self.curve)

    def test_path_tuple_construct(self):
        self.assertEqual(Path((self.xs, np.column_stack((self.sin, self.cos)))), self.path)

    def test_path_tuplelist_construct(self):
        self.assertEqual(Path([(self.xs, self.sin), (self.xs, self.cos)]), self.path)

    def test_path_ziplist_construct(self):
        self.assertEqual(Path([list(zip(self.xs, self.sin)), list(zip(self.xs, self.cos))]), self.path)

    def test_hist_zip_construct(self):
        self.assertEqual(Histogram(list(zip(self.hxs, self.sin))), self.histogram)

    def test_hist_array_construct(self):
        self.assertEqual(Histogram(np.column_stack((self.hxs, self.sin))), self.histogram)

    def test_hist_yvalues_construct(self):
        self.assertEqual(Histogram(self.sin), self.histogram)

    def test_hist_curve_construct(self):
        hist = Histogram(Curve(([0.1, 0.3, 0.5], [2.1, 2.2, 3.3])))
        values = hist.dimension_values(1)
        edges = hist.edges
        self.assertEqual(values, np.array([2.1, 2.2, 3.3]))
        self.assertEqual(edges, np.array([0, 0.2, 0.4, 0.6]))

    def test_hist_curve_int_edges_construct(self):
        hist = Histogram(Curve(range(3)))
        values = hist.dimension_values(1)
        edges = hist.edges
        self.assertEqual(values, np.arange(3))
        self.assertEqual(edges, np.array([-0.5, 0.5, 1.5, 2.5]))

    def test_heatmap_construct(self):
        hmap = HeatMap([('A', 'a', 1), ('B', 'b', 2)])
        dataset = Dataset({'x': ['A', 'B'], 'y': ['a', 'b'], 'z': [[1, np.nan], [np.nan, 2]]}, kdims=['x', 'y'], vdims=['z'], label='unique')
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_unsorted(self):
        hmap = HeatMap([('B', 'b', 2), ('A', 'a', 1)])
        dataset = Dataset({'x': ['B', 'A'], 'y': ['b', 'a'], 'z': [[2, np.nan], [np.nan, 1]]}, kdims=['x', 'y'], vdims=['z'], label='unique')
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_partial_sorted(self):
        data = [(chr(65 + i), chr(97 + j), i * j) for i in range(3) for j in [2, 0, 1] if i != j]
        hmap = HeatMap(data)
        dataset = Dataset({'x': ['A', 'B', 'C'], 'y': ['c', 'b', 'a'], 'z': [[0, 2, np.nan], [np.nan, 0, 0], [0, np.nan, 2]]}, kdims=['x', 'y'], vdims=['z'], label='unique')
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_and_sort(self):
        data = [(chr(65 + i), chr(97 + j), i * j) for i in range(3) for j in [2, 0, 1] if i != j]
        hmap = HeatMap(data).sort()
        dataset = Dataset({'x': ['A', 'B', 'C'], 'y': ['a', 'b', 'c'], 'z': [[np.nan, 0, 0], [0, np.nan, 2], [0, 2, np.nan]]}, kdims=['x', 'y'], vdims=['z'], label='unique')
        self.assertEqual(hmap.gridded, dataset)