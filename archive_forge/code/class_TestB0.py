import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import Nifti1Image
from .. import load as top_load
from ..optpkg import optional_package
from .nibabel_data import get_nibabel_data, needs_nibabel_data
class TestB0(TestEPIFrame):
    x_cos = [0.9970527523765, 0.0, 0.0767190261828617]
    y_cos = [0.0, 1.0, -6.9388939e-18]
    z_cos = [-0.0767190261828617, 6.9184432614435e-18, 0.9970527523765]
    zooms = [-0.8984375, -0.8984375, 6.49999990444107]
    starts = [105.473101260826, 151.74885125, -61.8714747993248]
    example_params = dict(fname=os.path.join(MINC2_PATH, 'mincex_diff-B0.mnc'), shape=(19, 256, 256), type=np.int16, affine=_make_affine((z_cos, y_cos, x_cos), zooms[::-1], starts[::-1]), zooms=[abs(v) for v in zooms[::-1]], min=4.566971917, max=3260.121093, mean=163.8305553)