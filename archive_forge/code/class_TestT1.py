import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import Nifti1Image
from .. import load as top_load
from ..optpkg import optional_package
from .nibabel_data import get_nibabel_data, needs_nibabel_data
class TestT1(TestEPIFrame):
    x_cos = [1, 0, 0]
    y_cos = [0, 1, 0]
    z_cos = [0, 0, 1]
    zooms = [1, 1, 1]
    starts = [-90, -126, -12]
    example_params = dict(fname=os.path.join(MINC2_PATH, 'mincex_t1.mnc'), shape=(110, 217, 181), type=np.int16, affine=_make_affine((z_cos, y_cos, x_cos), zooms[::-1], starts[::-1]), zooms=[abs(v) for v in zooms[::-1]], min=0, max=100, mean=23.1659928)