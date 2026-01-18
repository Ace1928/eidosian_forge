import os
import numpy as np
from numpy.testing import assert_array_equal
from .. import nifti2
from ..nifti1 import Nifti1Extension, Nifti1Extensions, Nifti1Header, Nifti1PairHeader
from ..nifti2 import Nifti2Header, Nifti2Image, Nifti2Pair, Nifti2PairHeader
from ..testing import data_path
from . import test_nifti1 as tn1
class _Nifti2Mixin:
    example_file = header_file
    sizeof_hdr = Nifti2Header.sizeof_hdr
    quat_dtype = np.float64

    def test_freesurfer_large_vector_hack(self):
        pass

    def test_freesurfer_ico7_hack(self):
        pass

    def test_eol_check(self):
        HC = self.header_class
        hdr = HC()
        good_eol = (13, 10, 26, 10)
        assert_array_equal(hdr['eol_check'], good_eol)
        hdr['eol_check'] = 0
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert_array_equal(fhdr['eol_check'], good_eol)
        assert message == 'EOL check all 0; setting EOL check to 13, 10, 26, 10'
        hdr['eol_check'] = (13, 10, 0, 10)
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert_array_equal(fhdr['eol_check'], good_eol)
        assert message == 'EOL check not 0 or 13, 10, 26, 10; data may be corrupted by EOL conversion; setting EOL check to 13, 10, 26, 10'