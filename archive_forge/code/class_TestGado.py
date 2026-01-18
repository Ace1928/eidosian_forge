import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import Nifti1Image
from .. import load as top_load
from ..optpkg import optional_package
from .nibabel_data import get_nibabel_data, needs_nibabel_data
class TestGado(TestEPIFrame):
    x_cos = [0.999695413509548, -0.0174524064372835, 0.0174497483512505]
    y_cos = [0.0174497483512505, 0.999847695156391, 0.000304586490452135]
    z_cos = [-0.0174524064372835, 0.0, 0.999847695156391]
    zooms = [1, -1, -1]
    starts = [-75.76775, 115.80462, 81.38605]
    example_params = dict(fname=os.path.join(MINC2_PATH, 'mincex_gado-contrast.mnc'), shape=(100, 170, 146), type=np.int16, affine=_make_affine((z_cos, y_cos, x_cos), zooms[::-1], starts[::-1]), zooms=[abs(v) for v in zooms[::-1]], min=0, max=938668.8698, mean=128169.3488)