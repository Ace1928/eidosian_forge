import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from ..ecat import load
from .nibabel_data import get_nibabel_data, needs_nibabel_data
class TestNegatives:
    opener = staticmethod(load)
    example_params = dict(fname=os.path.join(ECAT_TEST_PATH, 'ECAT7_testcaste_neg_values.v'), shape=(256, 256, 63, 1), type=np.int16, min=-0.00061576, max=0.19215, mean=0.04933)

    @needs_nibabel_data('nipy-ecattest')
    def test_load(self):
        img = self.opener(self.example_params['fname'])
        assert img.shape == self.example_params['shape']
        assert img.get_data_dtype(0).type == self.example_params['type']
        data = img.get_fdata()
        assert data.shape == self.example_params['shape']
        assert_almost_equal(data.min(), self.example_params['min'], 4)
        assert_almost_equal(data.max(), self.example_params['max'], 4)
        assert_almost_equal(data.mean(), self.example_params['mean'], 4)