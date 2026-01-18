import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
class GridTest(CompositeTest):

    def test_grid_init(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
        grid = GridSpace(zip(keys, vals))
        self.assertEqual(grid.shape, (2, 2))

    def test_grid_index_snap(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
        grid = GridSpace(zip(keys, vals))
        self.assertEqual(grid[0.1, 0.1], self.view1)

    def test_grid_index_strings(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [('A', 0), ('B', 1), ('C', 0), ('D', 1)]
        grid = GridSpace(zip(keys, vals))
        self.assertEqual(grid['B', 1], self.view2)

    def test_grid_index_one_axis(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [('A', 0), ('B', 1), ('C', 0), ('D', 1)]
        grid = GridSpace(zip(keys, vals))
        self.assertEqual(grid[:, 0], GridSpace([(('A', 0), self.view1), (('C', 0), self.view3)]))

    def test_gridspace_overlay_element(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = GridSpace(items, 'X')
        hline = HLine(0)
        overlaid_grid = grid * hline
        expected = GridSpace([(k, v * hline) for k, v in items], 'X')
        self.assertEqual(overlaid_grid, expected)

    def test_gridspace_overlay_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = GridSpace(items, 'X')
        hline = HLine(0)
        overlaid_grid = hline * grid
        expected = GridSpace([(k, hline * v) for k, v in items], 'X')
        self.assertEqual(overlaid_grid, expected)

    def test_gridspace_overlay_gridspace(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = GridSpace(items, 'X')
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = GridSpace(items2, 'X')
        expected_items = [(0, self.view1 * self.view2), (1, self.view2 * self.view1), (2, self.view3 * self.view2), (3, self.view2 * self.view3)]
        expected = GridSpace(expected_items, 'X')
        self.assertEqual(grid * grid2, expected)

    def test_gridspace_overlay_gridspace_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = GridSpace(items, 'X')
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = GridSpace(items2, 'X')
        expected_items = [(0, self.view2 * self.view1), (1, self.view1 * self.view2), (2, self.view2 * self.view3), (3, self.view3 * self.view2)]
        expected = GridSpace(expected_items, 'X')
        self.assertEqual(grid2 * grid, expected)

    def test_gridspace_overlay_gridspace_partial(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = GridSpace(items, 'X')
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = GridSpace(items2, 'X')
        expected_items = [(0, Overlay([self.view1, self.view2])), (1, Overlay([self.view2])), (2, Overlay([self.view2])), (3, Overlay([self.view2, self.view3]))]
        expected = GridSpace(expected_items, 'X')
        self.assertEqual(grid * grid2, expected)

    def test_gridspace_overlay_gridspace_partial_reverse(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = GridSpace(items, 'X')
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = GridSpace(items2, 'X')
        expected_items = [(0, Overlay([self.view2, self.view1])), (1, Overlay([self.view2])), (2, Overlay([self.view2])), (3, Overlay([self.view3, self.view2]))]
        expected = GridSpace(expected_items, 'X')
        self.assertEqual(grid2 * grid, expected)