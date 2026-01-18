import numpy as np
from skimage.measure import label
import skimage.measure._ccomp as ccomp
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
class TestSupport:

    def test_reshape(self):
        shapes_in = ((3, 1, 2), (1, 4, 5), (3, 1, 1), (2, 1), (1,))
        for shape in shapes_in:
            shape = np.array(shape)
            numones = sum(shape == 1)
            inp = np.random.random(shape)
            fixed, swaps = ccomp.reshape_array(inp)
            shape2 = fixed.shape
            for i in range(numones):
                assert shape2[i] == 1
            back = ccomp.undo_reshape_array(fixed, swaps)
            assert_array_equal(inp, back)