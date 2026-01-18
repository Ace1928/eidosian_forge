import copy
import operator
import sys
import unittest
import warnings
from collections import defaultdict
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...testing import assert_arrays_equal, clear_and_catch_warnings
from .. import tractogram as module_tractogram
from ..tractogram import (
class TestTractogram(unittest.TestCase):

    def test_tractogram_creation(self):
        tractogram = Tractogram()
        check_tractogram(tractogram)
        assert tractogram.affine_to_rasmm is None
        tractogram = Tractogram(streamlines=DATA['streamlines'])
        check_tractogram(tractogram, DATA['streamlines'])
        affine = np.diag([1, 2, 3, 1])
        tractogram = Tractogram(affine_to_rasmm=affine)
        assert_array_equal(tractogram.affine_to_rasmm, affine)
        tractogram = Tractogram(DATA['streamlines'], DATA['data_per_streamline'], DATA['data_per_point'])
        check_tractogram(tractogram, DATA['streamlines'], DATA['data_per_streamline'], DATA['data_per_point'])
        assert is_data_dict(tractogram.data_per_streamline)
        assert is_data_dict(tractogram.data_per_point)
        tractogram2 = Tractogram(tractogram.streamlines, tractogram.data_per_streamline, tractogram.data_per_point)
        assert_tractogram_equal(tractogram2, tractogram)
        tractogram = LazyTractogram(DATA['streamlines_func'], DATA['data_per_streamline_func'], DATA['data_per_point_func'])
        tractogram2 = Tractogram(tractogram.streamlines, tractogram.data_per_streamline, tractogram.data_per_point)
        wrong_data = [[(1, 0, 0)] * 1, [(0, 1, 0), (0, 1)], [(0, 0, 1)] * 5]
        data_per_point = {'wrong_data': wrong_data}
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)
        wrong_data = [[(1, 0, 0)] * 1, [(0, 1)] * 2, [(0, 0, 1)] * 5]
        data_per_point = {'wrong_data': wrong_data}
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)

    def test_setting_affine_to_rasmm(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.diag(range(4))
        tractogram.affine_to_rasmm = None
        assert tractogram.affine_to_rasmm is None
        tractogram.affine_to_rasmm = affine
        assert tractogram.affine_to_rasmm is not affine
        tractogram.affine_to_rasmm = affine.tolist()
        assert_array_equal(tractogram.affine_to_rasmm, affine)
        with pytest.raises(ValueError):
            tractogram.affine_to_rasmm = affine[::2]

    def test_tractogram_getitem(self):
        for i, t in enumerate(DATA['tractogram']):
            assert_tractogram_item_equal(DATA['tractogram'][i], t)
        tractogram_view = DATA['simple_tractogram'][::2]
        check_tractogram(tractogram_view, DATA['streamlines'][::2])
        r_tractogram = DATA['tractogram'][::-1]
        check_tractogram(r_tractogram, DATA['streamlines'][::-1], DATA['tractogram'].data_per_streamline[::-1], DATA['tractogram'].data_per_point[::-1])
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = DATA['rng'].rand(4, 4)
        tractogram_view = tractogram[::2]
        assert_array_equal(tractogram_view.affine_to_rasmm, tractogram.affine_to_rasmm)

    def test_tractogram_add_new_data(self):
        t = DATA['simple_tractogram'].copy()
        t.data_per_point['fa'] = DATA['fa']
        t.data_per_point['colors'] = DATA['colors']
        t.data_per_streamline['mean_curvature'] = DATA['mean_curvature']
        t.data_per_streamline['mean_torsion'] = DATA['mean_torsion']
        t.data_per_streamline['mean_colors'] = DATA['mean_colors']
        assert_tractogram_equal(t, DATA['tractogram'])
        for i, item in enumerate(t):
            assert_tractogram_item_equal(t[i], item)
        r_tractogram = t[::-1]
        check_tractogram(r_tractogram, t.streamlines[::-1], t.data_per_streamline[::-1], t.data_per_point[::-1])
        t = Tractogram(DATA['streamlines'] * 2, affine_to_rasmm=np.eye(4))
        t = t[:len(DATA['streamlines'])]
        t.data_per_point['fa'] = DATA['fa']
        t.data_per_point['colors'] = DATA['colors']
        t.data_per_streamline['mean_curvature'] = DATA['mean_curvature']
        t.data_per_streamline['mean_torsion'] = DATA['mean_torsion']
        t.data_per_streamline['mean_colors'] = DATA['mean_colors']
        assert_tractogram_equal(t, DATA['tractogram'])

    def test_tractogram_copy(self):
        tractogram = DATA['tractogram'].copy()
        assert tractogram is not DATA['tractogram']
        assert tractogram.streamlines is not DATA['tractogram'].streamlines
        assert tractogram.data_per_streamline is not DATA['tractogram'].data_per_streamline
        assert tractogram.data_per_point is not DATA['tractogram'].data_per_point
        for key in tractogram.data_per_streamline:
            assert tractogram.data_per_streamline[key] is not DATA['tractogram'].data_per_streamline[key]
        for key in tractogram.data_per_point:
            assert tractogram.data_per_point[key] is not DATA['tractogram'].data_per_point[key]
        assert_tractogram_equal(tractogram, DATA['tractogram'])

    def test_creating_invalid_tractogram(self):
        scalars = [[(1, 0, 0)] * 1, [(0, 1, 0)] * 2, [(0, 0, 1)] * 3]
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_point={'scalars': scalars})
        properties = [np.array([1.11, 1.22], dtype='f4'), np.array([3.11, 3.22], dtype='f4')]
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_streamline={'properties': properties})
        scalars = [[(1, 0, 0)] * 1, [(0, 1)] * 2, [(0, 0, 1)] * 5]
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_point={'scalars': scalars})
        properties = [[1.11, 1.22], [2.11], [3.11, 3.22]]
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_streamline={'properties': properties})
        properties = [np.array([[1.11], [1.22]], dtype='f4'), np.array([[2.11], [2.22]], dtype='f4'), np.array([[3.11], [3.22]], dtype='f4')]
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_streamline={'properties': properties})

    def test_tractogram_apply_affine(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.eye(4)
        scaling = np.array((1, 2, 3), dtype=float)
        affine[range(3), range(3)] = scaling
        transformed_tractogram = tractogram.apply_affine(affine, lazy=True)
        assert type(transformed_tractogram) is LazyTractogram
        check_tractogram(transformed_tractogram, streamlines=[s * scaling for s in DATA['streamlines']], data_per_streamline=DATA['data_per_streamline'], data_per_point=DATA['data_per_point'])
        assert_array_equal(transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.linalg.inv(affine)))
        assert_arrays_equal(tractogram.streamlines, DATA['streamlines'])
        transformed_tractogram = tractogram.apply_affine(affine)
        assert transformed_tractogram is tractogram
        check_tractogram(tractogram, streamlines=[s * scaling for s in DATA['streamlines']], data_per_streamline=DATA['data_per_streamline'], data_per_point=DATA['data_per_point'])
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.dot(np.linalg.inv(affine), np.linalg.inv(affine))))
        tractogram = DATA['tractogram'].copy()
        transformed_tractogram = tractogram[::2].apply_affine(affine)
        assert transformed_tractogram is not tractogram
        check_tractogram(tractogram[::2], streamlines=[s * scaling for s in DATA['streamlines'][::2]], data_per_streamline=DATA['tractogram'].data_per_streamline[::2], data_per_point=DATA['tractogram'].data_per_point[::2])
        check_tractogram(tractogram[1::2], streamlines=DATA['streamlines'][1::2], data_per_streamline=DATA['tractogram'].data_per_streamline[1::2], data_per_point=DATA['tractogram'].data_per_point[1::2])
        tractogram = DATA['tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]
        tractogram.apply_affine(affine)
        tractogram.apply_affine(np.linalg.inv(affine))
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)
        tractogram = DATA['tractogram'].copy()
        tractogram.apply_affine(np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = None
        tractogram.apply_affine(affine)
        assert tractogram.affine_to_rasmm is None

    def test_tractogram_to_world(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm, np.linalg.inv(affine))
        tractogram_world = transformed_tractogram.to_world(lazy=True)
        assert type(tractogram_world) is LazyTractogram
        assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)
        tractogram_world = transformed_tractogram.to_world()
        assert tractogram_world is tractogram
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)
        tractogram_world2 = transformed_tractogram.to_world()
        assert tractogram_world2 is tractogram
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = None
        with pytest.raises(ValueError):
            tractogram.to_world()

    def test_tractogram_extend(self):
        t = DATA['tractogram'].copy()
        for op, in_place in ((operator.add, False), (operator.iadd, True), (extender, True)):
            first_arg = t.copy()
            new_t = op(first_arg, t)
            assert (new_t is first_arg) == in_place
            assert_tractogram_equal(new_t[:len(t)], DATA['tractogram'])
            assert_tractogram_equal(new_t[len(t):], DATA['tractogram'])
        t = Tractogram()
        t += DATA['tractogram']
        assert_tractogram_equal(t, DATA['tractogram'])
        t = DATA['tractogram'].copy()
        t += Tractogram()
        assert_tractogram_equal(t, DATA['tractogram'])