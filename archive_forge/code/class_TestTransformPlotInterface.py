import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
class TestTransformPlotInterface:

    def test_line_extent_axes_coords(self):
        ax = plt.axes()
        ax.plot([0.1, 1.2, 0.8], [0.9, 0.5, 0.8], transform=ax.transAxes)
        assert_array_equal(ax.dataLim.get_points(), np.array([[np.inf, np.inf], [-np.inf, -np.inf]]))

    def test_line_extent_data_coords(self):
        ax = plt.axes()
        ax.plot([0.1, 1.2, 0.8], [0.9, 0.5, 0.8], transform=ax.transData)
        assert_array_equal(ax.dataLim.get_points(), np.array([[0.1, 0.5], [1.2, 0.9]]))

    def test_line_extent_compound_coords1(self):
        ax = plt.axes()
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.plot([0.1, 1.2, 0.8], [35, -5, 18], transform=trans)
        assert_array_equal(ax.dataLim.get_points(), np.array([[np.inf, -5.0], [-np.inf, 35.0]]))

    def test_line_extent_predata_transform_coords(self):
        ax = plt.axes()
        trans = mtransforms.Affine2D().scale(10) + ax.transData
        ax.plot([0.1, 1.2, 0.8], [35, -5, 18], transform=trans)
        assert_array_equal(ax.dataLim.get_points(), np.array([[1.0, -50.0], [12.0, 350.0]]))

    def test_line_extent_compound_coords2(self):
        ax = plt.axes()
        trans = mtransforms.blended_transform_factory(ax.transAxes, mtransforms.Affine2D().scale(10) + ax.transData)
        ax.plot([0.1, 1.2, 0.8], [35, -5, 18], transform=trans)
        assert_array_equal(ax.dataLim.get_points(), np.array([[np.inf, -50.0], [-np.inf, 350.0]]))

    def test_line_extents_affine(self):
        ax = plt.axes()
        offset = mtransforms.Affine2D().translate(10, 10)
        plt.plot(np.arange(10), transform=offset + ax.transData)
        expected_data_lim = np.array([[0.0, 0.0], [9.0, 9.0]]) + 10
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)

    def test_line_extents_non_affine(self):
        ax = plt.axes()
        offset = mtransforms.Affine2D().translate(10, 10)
        na_offset = NonAffineForTest(mtransforms.Affine2D().translate(10, 10))
        plt.plot(np.arange(10), transform=offset + na_offset + ax.transData)
        expected_data_lim = np.array([[0.0, 0.0], [9.0, 9.0]]) + 20
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)

    def test_pathc_extents_non_affine(self):
        ax = plt.axes()
        offset = mtransforms.Affine2D().translate(10, 10)
        na_offset = NonAffineForTest(mtransforms.Affine2D().translate(10, 10))
        pth = Path([[0, 0], [0, 10], [10, 10], [10, 0]])
        patch = mpatches.PathPatch(pth, transform=offset + na_offset + ax.transData)
        ax.add_patch(patch)
        expected_data_lim = np.array([[0.0, 0.0], [10.0, 10.0]]) + 20
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)

    def test_pathc_extents_affine(self):
        ax = plt.axes()
        offset = mtransforms.Affine2D().translate(10, 10)
        pth = Path([[0, 0], [0, 10], [10, 10], [10, 0]])
        patch = mpatches.PathPatch(pth, transform=offset + ax.transData)
        ax.add_patch(patch)
        expected_data_lim = np.array([[0.0, 0.0], [10.0, 10.0]]) + 10
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)

    def test_line_extents_for_non_affine_transData(self):
        ax = plt.axes(projection='polar')
        offset = mtransforms.Affine2D().translate(0, 10)
        plt.plot(np.arange(10), transform=offset + ax.transData)
        expected_data_lim = np.array([[0.0, 0.0], [9.0, 9.0]]) + [0, 10]
        assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)