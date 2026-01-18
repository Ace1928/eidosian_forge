import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import Nifti1Image
from .. import load as top_load
from ..optpkg import optional_package
from .nibabel_data import get_nibabel_data, needs_nibabel_data
class TestMask(TestEPIFrame):
    example_params = TestT1.example_params.copy()
    new_params = dict(fname=os.path.join(MINC2_PATH, 'mincex_mask.mnc'), type=np.uint8, min=0, max=1, mean=0.3817466618)
    example_params.update(new_params)