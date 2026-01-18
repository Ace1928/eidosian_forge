import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
class TestDendrogram:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_dendrogram_single_linkage_tdist(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        R = dendrogram(Z, no_plot=True)
        leaves = R['leaves']
        assert_equal(leaves, [2, 5, 1, 0, 3, 4])

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_valid_orientation(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        assert_raises(ValueError, dendrogram, Z, orientation='foo')

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_labels_as_array_or_list(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        labels = xp.asarray([1, 3, 2, 6, 4, 5])
        result1 = dendrogram(Z, labels=labels, no_plot=True)
        result2 = dendrogram(Z, labels=list(labels), no_plot=True)
        assert result1 == result2

    @skip_if_array_api_gpu
    @array_api_compatible
    @pytest.mark.skipif(not have_matplotlib, reason='no matplotlib')
    def test_valid_label_size(self, xp):
        link = xp.asarray([[0, 1, 1.0, 4], [2, 3, 1.0, 5], [4, 5, 2.0, 6]])
        plt.figure()
        with pytest.raises(ValueError) as exc_info:
            dendrogram(link, labels=list(range(100)))
        assert 'Dimensions of Z and labels must be consistent.' in str(exc_info.value)
        with pytest.raises(ValueError, match='Dimensions of Z and labels must be consistent.'):
            dendrogram(link, labels=[])
        plt.close()

    @skip_if_array_api_gpu
    @array_api_compatible
    @pytest.mark.skipif(not have_matplotlib, reason='no matplotlib')
    def test_dendrogram_plot(self, xp):
        for orientation in ['top', 'bottom', 'left', 'right']:
            self.check_dendrogram_plot(orientation, xp)

    def check_dendrogram_plot(self, orientation, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        expected = {'color_list': ['C1', 'C0', 'C0', 'C0', 'C0'], 'dcoord': [[0.0, 138.0, 138.0, 0.0], [0.0, 219.0, 219.0, 0.0], [0.0, 255.0, 255.0, 219.0], [0.0, 268.0, 268.0, 255.0], [138.0, 295.0, 295.0, 268.0]], 'icoord': [[5.0, 5.0, 15.0, 15.0], [45.0, 45.0, 55.0, 55.0], [35.0, 35.0, 50.0, 50.0], [25.0, 25.0, 42.5, 42.5], [10.0, 10.0, 33.75, 33.75]], 'ivl': ['2', '5', '1', '0', '3', '4'], 'leaves': [2, 5, 1, 0, 3, 4], 'leaves_color_list': ['C1', 'C1', 'C0', 'C0', 'C0', 'C0']}
        fig = plt.figure()
        ax = fig.add_subplot(221)
        R1 = dendrogram(Z, ax=ax, orientation=orientation)
        R1['dcoord'] = np.asarray(R1['dcoord'])
        assert_equal(R1, expected)
        dendrogram(Z, ax=ax, orientation=orientation, leaf_font_size=20, leaf_rotation=90)
        testlabel = ax.get_xticklabels()[0] if orientation in ['top', 'bottom'] else ax.get_yticklabels()[0]
        assert_equal(testlabel.get_rotation(), 90)
        assert_equal(testlabel.get_size(), 20)
        dendrogram(Z, ax=ax, orientation=orientation, leaf_rotation=90)
        testlabel = ax.get_xticklabels()[0] if orientation in ['top', 'bottom'] else ax.get_yticklabels()[0]
        assert_equal(testlabel.get_rotation(), 90)
        dendrogram(Z, ax=ax, orientation=orientation, leaf_font_size=20)
        testlabel = ax.get_xticklabels()[0] if orientation in ['top', 'bottom'] else ax.get_yticklabels()[0]
        assert_equal(testlabel.get_size(), 20)
        plt.close()
        R2 = dendrogram(Z, orientation=orientation)
        plt.close()
        R2['dcoord'] = np.asarray(R2['dcoord'])
        assert_equal(R2, expected)

    @skip_if_array_api_gpu
    @array_api_compatible
    @pytest.mark.skipif(not have_matplotlib, reason='no matplotlib')
    def test_dendrogram_truncate_mode(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        R = dendrogram(Z, 2, 'lastp', show_contracted=True)
        plt.close()
        R['dcoord'] = np.asarray(R['dcoord'])
        assert_equal(R, {'color_list': ['C0'], 'dcoord': [[0.0, 295.0, 295.0, 0.0]], 'icoord': [[5.0, 5.0, 15.0, 15.0]], 'ivl': ['(2)', '(4)'], 'leaves': [6, 9], 'leaves_color_list': ['C0', 'C0']})
        R = dendrogram(Z, 2, 'mtica', show_contracted=True)
        plt.close()
        R['dcoord'] = np.asarray(R['dcoord'])
        assert_equal(R, {'color_list': ['C1', 'C0', 'C0', 'C0'], 'dcoord': [[0.0, 138.0, 138.0, 0.0], [0.0, 255.0, 255.0, 0.0], [0.0, 268.0, 268.0, 255.0], [138.0, 295.0, 295.0, 268.0]], 'icoord': [[5.0, 5.0, 15.0, 15.0], [35.0, 35.0, 45.0, 45.0], [25.0, 25.0, 40.0, 40.0], [10.0, 10.0, 32.5, 32.5]], 'ivl': ['2', '5', '1', '0', '(2)'], 'leaves': [2, 5, 1, 0, 7], 'leaves_color_list': ['C1', 'C1', 'C0', 'C0', 'C0']})

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_dendrogram_colors(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        set_link_color_palette(['c', 'm', 'y', 'k'])
        R = dendrogram(Z, no_plot=True, above_threshold_color='g', color_threshold=250)
        set_link_color_palette(['g', 'r', 'c', 'm', 'y', 'k'])
        color_list = R['color_list']
        assert_equal(color_list, ['c', 'm', 'g', 'g', 'g'])
        set_link_color_palette(None)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_dendrogram_leaf_colors_zero_dist(self, xp):
        x = xp.asarray([[1, 0, 0], [0, 0, 1], [0, 2, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
        z = linkage(x, 'single')
        d = dendrogram(z, no_plot=True)
        exp_colors = ['C0', 'C1', 'C1', 'C0', 'C2', 'C2']
        colors = d['leaves_color_list']
        assert_equal(colors, exp_colors)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_dendrogram_leaf_colors(self, xp):
        x = xp.asarray([[1, 0, 0], [0, 0, 1.1], [0, 2, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
        z = linkage(x, 'single')
        d = dendrogram(z, no_plot=True)
        exp_colors = ['C0', 'C1', 'C1', 'C0', 'C2', 'C2']
        colors = d['leaves_color_list']
        assert_equal(colors, exp_colors)