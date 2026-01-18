import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
def _check_maps(ar, ar_relab, fw, inv):
    assert_array_equal(fw[ar], ar_relab)
    assert_array_equal(inv[ar_relab], ar)