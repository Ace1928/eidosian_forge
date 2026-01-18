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
class TestPerArrayDict(unittest.TestCase):

    def test_per_array_dict_creation(self):
        nb_streamlines = len(DATA['tractogram'])
        data_per_streamline = DATA['tractogram'].data_per_streamline
        data_dict = PerArrayDict(nb_streamlines, data_per_streamline)
        assert data_dict.keys() == data_per_streamline.keys()
        for k in data_dict.keys():
            assert_array_equal(data_dict[k], data_per_streamline[k])
        del data_dict['mean_curvature']
        assert len(data_dict) == len(data_per_streamline) - 1
        data_per_streamline = DATA['data_per_streamline']
        data_dict = PerArrayDict(nb_streamlines, data_per_streamline)
        assert data_dict.keys() == data_per_streamline.keys()
        for k in data_dict.keys():
            assert_array_equal(data_dict[k], data_per_streamline[k])
        del data_dict['mean_curvature']
        assert len(data_dict) == len(data_per_streamline) - 1
        data_per_streamline = DATA['data_per_streamline']
        data_dict = PerArrayDict(nb_streamlines, **data_per_streamline)
        assert data_dict.keys() == data_per_streamline.keys()
        for k in data_dict.keys():
            assert_array_equal(data_dict[k], data_per_streamline[k])
        del data_dict['mean_curvature']
        assert len(data_dict) == len(data_per_streamline) - 1

    def test_getitem(self):
        sdict = PerArrayDict(len(DATA['tractogram']), DATA['data_per_streamline'])
        with pytest.raises(KeyError):
            sdict['invalid']
        for k, v in DATA['tractogram'].data_per_streamline.items():
            assert k in sdict
            assert_arrays_equal(sdict[k], v)
            assert_arrays_equal(sdict[::2][k], v[::2])
            assert_arrays_equal(sdict[::-1][k], v[::-1])
            assert_arrays_equal(sdict[-1][k], v[-1])
            assert_arrays_equal(sdict[[0, -1]][k], v[[0, -1]])

    def test_extend(self):
        sdict = PerArrayDict(len(DATA['tractogram']), DATA['data_per_streamline'])
        new_data = {'mean_curvature': 2 * np.array(DATA['mean_curvature']), 'mean_torsion': 3 * np.array(DATA['mean_torsion']), 'mean_colors': 4 * np.array(DATA['mean_colors'])}
        sdict2 = PerArrayDict(len(DATA['tractogram']), new_data)
        sdict.extend(sdict2)
        assert len(sdict) == len(sdict2)
        for k in DATA['tractogram'].data_per_streamline:
            assert_arrays_equal(sdict[k][:len(DATA['tractogram'])], DATA['tractogram'].data_per_streamline[k])
            assert_arrays_equal(sdict[k][len(DATA['tractogram']):], new_data[k])
        sdict_orig = copy.deepcopy(sdict)
        sdict.extend(PerArrayDict())
        for k in sdict_orig.keys():
            assert_arrays_equal(sdict[k], sdict_orig[k])
        new_data = {'mean_curvature': 2 * np.array(DATA['mean_curvature']), 'mean_torsion': 3 * np.array(DATA['mean_torsion']), 'mean_colors': 4 * np.array(DATA['mean_colors']), 'other': 5 * np.array(DATA['mean_colors'])}
        sdict2 = PerArrayDict(len(DATA['tractogram']), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)
        new_data = {'mean_curvature': 2 * np.array(DATA['mean_curvature']), 'mean_torsion': 3 * np.array(DATA['mean_torsion']), 'other': 4 * np.array(DATA['mean_colors'])}
        sdict2 = PerArrayDict(len(DATA['tractogram']), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)
        new_data = {'mean_curvature': 2 * np.array(DATA['mean_curvature']), 'mean_torsion': 3 * np.array(DATA['mean_torsion']), 'mean_colors': 4 * np.array(DATA['mean_torsion'])}
        sdict2 = PerArrayDict(len(DATA['tractogram']), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)