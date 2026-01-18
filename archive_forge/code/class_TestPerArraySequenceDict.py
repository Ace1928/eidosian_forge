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
class TestPerArraySequenceDict(unittest.TestCase):

    def test_per_array_sequence_dict_creation(self):
        total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
        data_per_point = DATA['tractogram'].data_per_point
        data_dict = PerArraySequenceDict(total_nb_rows, data_per_point)
        assert data_dict.keys() == data_per_point.keys()
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])
        del data_dict['fa']
        assert len(data_dict) == len(data_per_point) - 1
        data_per_point = DATA['data_per_point']
        data_dict = PerArraySequenceDict(total_nb_rows, data_per_point)
        assert data_dict.keys() == data_per_point.keys()
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])
        del data_dict['fa']
        assert len(data_dict) == len(data_per_point) - 1
        data_per_point = DATA['data_per_point']
        data_dict = PerArraySequenceDict(total_nb_rows, **data_per_point)
        assert data_dict.keys() == data_per_point.keys()
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])
        del data_dict['fa']
        assert len(data_dict) == len(data_per_point) - 1

    def test_getitem(self):
        total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
        sdict = PerArraySequenceDict(total_nb_rows, DATA['data_per_point'])
        with pytest.raises(KeyError):
            sdict['invalid']
        for k, v in DATA['tractogram'].data_per_point.items():
            assert k in sdict
            assert_arrays_equal(sdict[k], v)
            assert_arrays_equal(sdict[::2][k], v[::2])
            assert_arrays_equal(sdict[::-1][k], v[::-1])
            assert_arrays_equal(sdict[-1][k], v[-1])
            assert_arrays_equal(sdict[[0, -1]][k], v[[0, -1]])

    def test_extend(self):
        total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
        sdict = PerArraySequenceDict(total_nb_rows, DATA['data_per_point'])
        list_nb_points = [2, 7, 4]
        data_per_point_shapes = {'colors': DATA['colors'][0].shape[1:], 'fa': DATA['fa'][0].shape[1:]}
        _, new_data, _ = make_fake_tractogram(list_nb_points, data_per_point_shapes, rng=DATA['rng'])
        sdict2 = PerArraySequenceDict(np.sum(list_nb_points), new_data)
        sdict.extend(sdict2)
        assert len(sdict) == len(sdict2)
        for k in DATA['tractogram'].data_per_point:
            assert_arrays_equal(sdict[k][:len(DATA['tractogram'])], DATA['tractogram'].data_per_point[k])
            assert_arrays_equal(sdict[k][len(DATA['tractogram']):], new_data[k])
        sdict_orig = copy.deepcopy(sdict)
        sdict.extend(PerArraySequenceDict())
        for k in sdict_orig.keys():
            assert_arrays_equal(sdict[k], sdict_orig[k])
        data_per_point_shapes = {'colors': DATA['colors'][0].shape[1:], 'fa': DATA['fa'][0].shape[1:], 'other': (7,)}
        _, new_data, _ = make_fake_tractogram(list_nb_points, data_per_point_shapes, rng=DATA['rng'])
        sdict2 = PerArraySequenceDict(np.sum(list_nb_points), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)
        data_per_point_shapes = {'colors': DATA['colors'][0].shape[1:], 'other': DATA['fa'][0].shape[1:]}
        _, new_data, _ = make_fake_tractogram(list_nb_points, data_per_point_shapes, rng=DATA['rng'])
        sdict2 = PerArraySequenceDict(np.sum(list_nb_points), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)
        data_per_point_shapes = {'colors': DATA['colors'][0].shape[1:], 'fa': DATA['fa'][0].shape[1:] + (3,)}
        _, new_data, _ = make_fake_tractogram(list_nb_points, data_per_point_shapes, rng=DATA['rng'])
        sdict2 = PerArraySequenceDict(np.sum(list_nb_points), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)