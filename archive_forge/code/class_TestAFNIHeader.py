from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, brikhead, load
from ..testing import assert_data_similar, data_path
from .test_fileslice import slicer_samples
class TestAFNIHeader:
    module = brikhead
    test_files = EXAMPLE_IMAGES

    def test_makehead(self):
        for tp in self.test_files:
            head1 = self.module.AFNIHeader.from_fileobj(tp['head'])
            head2 = self.module.AFNIHeader.from_header(head1)
            assert head1 == head2
            with pytest.raises(self.module.AFNIHeaderError):
                self.module.AFNIHeader.from_header(header=None)
            with pytest.raises(self.module.AFNIHeaderError):
                self.module.AFNIHeader.from_header(tp['fname'])