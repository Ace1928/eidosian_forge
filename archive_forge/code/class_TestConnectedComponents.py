import numpy as np
from skimage.measure import label
import skimage.measure._ccomp as ccomp
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
class TestConnectedComponents:

    def setup_method(self):
        self.x = np.array([[0, 0, 3, 2, 1, 9], [0, 1, 1, 9, 2, 9], [0, 0, 1, 9, 9, 9], [3, 1, 1, 5, 3, 0]])
        self.labels = np.array([[0, 0, 1, 2, 3, 4], [0, 5, 5, 4, 2, 4], [0, 0, 5, 4, 4, 4], [6, 5, 5, 7, 8, 0]])
        self.labels_nobg = self.labels + 1
        self.labels_nobg[-1, -1] = 10
        self.labels_bg_9 = self.labels_nobg.copy()
        self.labels_bg_9[self.x == 9] = 0
        self.labels_bg_9[self.labels_bg_9 > 5] -= 1

    def test_basic(self):
        assert_array_equal(label(self.x), self.labels)
        assert self.x[0, 2] == 3
        assert_array_equal(label(self.x, background=99), self.labels_nobg)
        assert_array_equal(label(self.x, background=9), self.labels_bg_9)

    def test_random(self):
        x = (np.random.rand(20, 30) * 5).astype(int)
        labels = label(x)
        n = labels.max()
        for i in range(n):
            values = x[labels == i]
            assert np.all(values == values[0])

    def test_diag(self):
        x = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        assert_array_equal(label(x), x)

    def test_4_vs_8(self):
        x = np.array([[0, 1], [1, 0]], dtype=int)
        assert_array_equal(label(x, connectivity=1), [[0, 1], [2, 0]])
        assert_array_equal(label(x, connectivity=2), [[0, 1], [1, 0]])

    def test_background(self):
        x = np.array([[1, 0, 0], [1, 1, 5], [0, 0, 0]])
        assert_array_equal(label(x), [[1, 0, 0], [1, 1, 2], [0, 0, 0]])
        assert_array_equal(label(x, background=0), [[1, 0, 0], [1, 1, 2], [0, 0, 0]])

    def test_background_two_regions(self):
        x = np.array([[0, 0, 6], [0, 0, 6], [5, 5, 5]])
        res = label(x, background=0)
        assert_array_equal(res, [[0, 0, 1], [0, 0, 1], [2, 2, 2]])

    def test_background_one_region_center(self):
        x = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_array_equal(label(x, connectivity=1, background=0), [[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    def test_return_num(self):
        x = np.array([[1, 0, 6], [0, 0, 6], [5, 5, 5]])
        assert_array_equal(label(x, return_num=True)[1], 3)
        assert_array_equal(label(x, background=-1, return_num=True)[1], 4)