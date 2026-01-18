import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
class TestConditionalShortcuts:

    @pytest.mark.parametrize('mode', _all_modes.keys())
    def test_zero_padding_shortcuts(self, mode):
        test = np.arange(120).reshape(4, 5, 6)
        pad_amt = [(0, 0) for _ in test.shape]
        assert_array_equal(test, np.pad(test, pad_amt, mode=mode))

    @pytest.mark.parametrize('mode', ['maximum', 'mean', 'median', 'minimum'])
    def test_shallow_statistic_range(self, mode):
        test = np.arange(120).reshape(4, 5, 6)
        pad_amt = [(1, 1) for _ in test.shape]
        assert_array_equal(np.pad(test, pad_amt, mode='edge'), np.pad(test, pad_amt, mode=mode, stat_length=1))

    @pytest.mark.parametrize('mode', ['maximum', 'mean', 'median', 'minimum'])
    def test_clip_statistic_range(self, mode):
        test = np.arange(30).reshape(5, 6)
        pad_amt = [(3, 3) for _ in test.shape]
        assert_array_equal(np.pad(test, pad_amt, mode=mode), np.pad(test, pad_amt, mode=mode, stat_length=30))