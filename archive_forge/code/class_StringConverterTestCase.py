import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
class StringConverterTestCase:
    allow_bytes = True
    case_insensitive = True
    exact_match = False
    warn = True

    def _check_value_error(self, val):
        pattern = '\\(got {}\\)'.format(re.escape(repr(val)))
        with pytest.raises(ValueError, match=pattern) as exc:
            self.conv(val)

    def _check_conv_assert_warn(self, val, expected):
        if self.warn:
            with assert_warns(DeprecationWarning) as exc:
                assert self.conv(val) == expected
        else:
            assert self.conv(val) == expected

    def _check(self, val, expected):
        """Takes valid non-deprecated inputs for converters,
        runs converters on inputs, checks correctness of outputs,
        warnings and errors"""
        assert self.conv(val) == expected
        if self.allow_bytes:
            assert self.conv(val.encode('ascii')) == expected
        else:
            with pytest.raises(TypeError):
                self.conv(val.encode('ascii'))
        if len(val) != 1:
            if self.exact_match:
                self._check_value_error(val[:1])
                self._check_value_error(val + '\x00')
            else:
                self._check_conv_assert_warn(val[:1], expected)
        if self.case_insensitive:
            if val != val.lower():
                self._check_conv_assert_warn(val.lower(), expected)
            if val != val.upper():
                self._check_conv_assert_warn(val.upper(), expected)
        else:
            if val != val.lower():
                self._check_value_error(val.lower())
            if val != val.upper():
                self._check_value_error(val.upper())

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            self.conv({})
        with pytest.raises(TypeError):
            self.conv([])

    def test_wrong_value(self):
        self._check_value_error('')
        self._check_value_error('π')
        if self.allow_bytes:
            self._check_value_error(b'')
            self._check_value_error(b'\xff')
        if self.exact_match:
            self._check_value_error("there's no way this is supported")