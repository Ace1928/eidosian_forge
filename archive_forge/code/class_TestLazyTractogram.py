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
class TestLazyTractogram(unittest.TestCase):

    def test_lazy_tractogram_creation(self):
        with pytest.raises(TypeError):
            LazyTractogram(streamlines=DATA['streamlines'])
        streamlines = (x for x in DATA['streamlines'])
        data_per_point = {'colors': (x for x in DATA['colors'])}
        data_per_streamline = {'torsion': (x for x in DATA['mean_torsion']), 'colors': (x for x in DATA['mean_colors'])}
        with pytest.raises(TypeError):
            LazyTractogram(streamlines=streamlines)
        with pytest.raises(TypeError):
            LazyTractogram(data_per_point={'none': None})
        with pytest.raises(TypeError):
            LazyTractogram(data_per_streamline=data_per_streamline)
        with pytest.raises(TypeError):
            LazyTractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)
        tractogram = LazyTractogram()
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            check_tractogram(tractogram)
        assert tractogram.affine_to_rasmm is None
        tractogram = LazyTractogram(DATA['streamlines_func'], DATA['data_per_streamline_func'], DATA['data_per_point_func'])
        assert is_lazy_dict(tractogram.data_per_streamline)
        assert is_lazy_dict(tractogram.data_per_point)
        [t for t in tractogram]
        assert len(tractogram) == len(DATA['streamlines'])
        for i in range(2):
            assert_tractogram_equal(tractogram, DATA['tractogram'])

    def test_lazy_tractogram_from_data_func(self):
        tractogram = LazyTractogram.from_data_func(lambda: iter([]))
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            check_tractogram(tractogram)
        data = [DATA['streamlines'], DATA['fa'], DATA['colors'], DATA['mean_curvature'], DATA['mean_torsion'], DATA['mean_colors']]

        def _data_gen():
            for d in zip(*data):
                data_for_points = {'fa': d[1], 'colors': d[2]}
                data_for_streamline = {'mean_curvature': d[3], 'mean_torsion': d[4], 'mean_colors': d[5]}
                yield TractogramItem(d[0], data_for_streamline, data_for_points)
        tractogram = LazyTractogram.from_data_func(_data_gen)
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            assert_tractogram_equal(tractogram, DATA['tractogram'])
        with pytest.raises(TypeError):
            LazyTractogram.from_data_func(_data_gen())

    def test_lazy_tractogram_getitem(self):
        with pytest.raises(NotImplementedError):
            DATA['lazy_tractogram'][0]

    def test_lazy_tractogram_extend(self):
        t = DATA['lazy_tractogram'].copy()
        new_t = DATA['lazy_tractogram'].copy()
        for op in (operator.add, operator.iadd, extender):
            with pytest.raises(NotImplementedError):
                op(new_t, t)

    def test_lazy_tractogram_len(self):
        modules = [module_tractogram]
        with clear_and_catch_warnings(record=True, modules=modules) as w:
            warnings.simplefilter('always')
            tractogram = LazyTractogram(DATA['streamlines_func'])
            assert tractogram._nb_streamlines is None
            assert len(tractogram) == len(DATA['streamlines'])
            assert tractogram._nb_streamlines == len(DATA['streamlines'])
            assert len(w) == 1
            tractogram = LazyTractogram(DATA['streamlines_func'])
            assert len(tractogram) == len(DATA['streamlines'])
            assert len(w) == 2
            assert issubclass(w[-1].category, Warning) is True
            assert len(tractogram) == len(DATA['streamlines'])
            assert len(w) == 2
        with clear_and_catch_warnings(record=True, modules=modules) as w:
            tractogram = LazyTractogram(DATA['streamlines_func'])
            assert tractogram._nb_streamlines is None
            [t for t in tractogram]
            assert tractogram._nb_streamlines == len(DATA['streamlines'])
            assert len(tractogram) == len(DATA['streamlines'])
            assert len(w) == 0

    def test_lazy_tractogram_apply_affine(self):
        affine = np.eye(4)
        scaling = np.array((1, 2, 3), dtype=float)
        affine[range(3), range(3)] = scaling
        tractogram = DATA['lazy_tractogram'].copy()
        transformed_tractogram = tractogram.apply_affine(affine)
        assert transformed_tractogram is not tractogram
        assert_array_equal(tractogram._affine_to_apply, np.eye(4))
        assert_array_equal(tractogram.affine_to_rasmm, np.eye(4))
        assert_array_equal(transformed_tractogram._affine_to_apply, affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.linalg.inv(affine)))
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            check_tractogram(transformed_tractogram, streamlines=[s * scaling for s in DATA['streamlines']], data_per_streamline=DATA['data_per_streamline'], data_per_point=DATA['data_per_point'])
        transformed_tractogram = transformed_tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram._affine_to_apply, np.dot(affine, affine))
        assert_array_equal(transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.dot(np.linalg.inv(affine), np.linalg.inv(affine))))
        tractogram = DATA['lazy_tractogram'].copy()
        tractogram.affine_to_rasmm = None
        with pytest.raises(ValueError):
            tractogram.to_world()
        tractogram = DATA['lazy_tractogram'].copy()
        tractogram.affine_to_rasmm = None
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram._affine_to_apply, affine)
        assert transformed_tractogram.affine_to_rasmm is None
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            check_tractogram(transformed_tractogram, streamlines=[s * scaling for s in DATA['streamlines']], data_per_streamline=DATA['data_per_streamline'], data_per_point=DATA['data_per_point'])
        tractogram = DATA['lazy_tractogram'].copy()
        with pytest.raises(ValueError):
            tractogram.apply_affine(affine=np.eye(4), lazy=False)

    def test_tractogram_to_world(self):
        tractogram = DATA['lazy_tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm, np.linalg.inv(affine))
        tractogram_world = transformed_tractogram.to_world()
        assert tractogram_world is not transformed_tractogram
        assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)
        tractogram_world = tractogram_world.to_world()
        assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)
        tractogram = DATA['lazy_tractogram'].copy()
        tractogram.affine_to_rasmm = None
        with pytest.raises(ValueError):
            tractogram.to_world()

    def test_lazy_tractogram_copy(self):
        tractogram = DATA['lazy_tractogram'].copy()
        assert tractogram is not DATA['lazy_tractogram']
        assert tractogram._streamlines is DATA['lazy_tractogram']._streamlines
        assert tractogram._data_per_streamline is not DATA['lazy_tractogram']._data_per_streamline
        assert tractogram._data_per_point is not DATA['lazy_tractogram']._data_per_point
        for key in tractogram.data_per_streamline:
            data = tractogram.data_per_streamline.store[key]
            expected = DATA['lazy_tractogram'].data_per_streamline.store[key]
            assert data is expected
        for key in tractogram.data_per_point:
            data = tractogram.data_per_point.store[key]
            expected = DATA['lazy_tractogram'].data_per_point.store[key]
            assert data is expected
        assert tractogram._affine_to_apply is not DATA['lazy_tractogram']._affine_to_apply
        assert_array_equal(tractogram._affine_to_apply, DATA['lazy_tractogram']._affine_to_apply)
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            assert_tractogram_equal(tractogram, DATA['tractogram'])