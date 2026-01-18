import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
class TestMBLBP:

    def test_single_mblbp(self):
        test_img = np.zeros((9, 9), dtype='uint8')
        test_img[3:6, 3:6] = 1
        test_img[:3, :3] = 255
        test_img[6:, 6:] = 255
        correct_answer = 136
        int_img = integral_image(test_img)
        lbp_code = multiblock_lbp(int_img, 0, 0, 3, 3)
        np.testing.assert_equal(lbp_code, correct_answer)