import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
class TestChanges:
    """
    These test cases exercise some behaviour changes
    """

    @pytest.mark.parametrize('string', ['S', 'U'])
    @pytest.mark.parametrize('floating', ['e', 'f', 'd', 'g'])
    def test_float_to_string(self, floating, string):
        assert np.can_cast(floating, string)
        assert np.can_cast(floating, f'{string}100')

    def test_to_void(self):
        assert np.can_cast('d', 'V')
        assert np.can_cast('S20', 'V')
        assert not np.can_cast('d', 'V1')
        assert not np.can_cast('S20', 'V1')
        assert not np.can_cast('U1', 'V1')
        assert np.can_cast('d,i', 'V', casting='same_kind')
        assert np.can_cast('V3', 'V', casting='no')
        assert np.can_cast('V0', 'V', casting='no')