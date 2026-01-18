from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, brikhead, load
from ..testing import assert_data_similar, data_path
from .test_fileslice import slicer_samples
class TestBadVars:
    module = brikhead
    vars = ['type = badtype-attribute\nname = BRICK_TYPES\ncount = 1\n1\n', 'type = integer-attribute\ncount = 1\n1\n']

    def test_unpack_var(self):
        for var in self.vars:
            with pytest.raises(self.module.AFNIHeaderError):
                self.module._unpack_var(var)