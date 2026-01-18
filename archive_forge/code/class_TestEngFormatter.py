from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
class TestEngFormatter:
    raw_format_data = [(False, -1234.56789, ('-1.23457 k', '-1 k', '-1.23 k')), (True, -1234.56789, ('−1.23457 k', '−1 k', '−1.23 k')), (False, -1.23456789, ('-1.23457', '-1', '-1.23')), (True, -1.23456789, ('−1.23457', '−1', '−1.23')), (False, -0.123456789, ('-123.457 m', '-123 m', '-123.46 m')), (True, -0.123456789, ('−123.457 m', '−123 m', '−123.46 m')), (False, -0.00123456789, ('-1.23457 m', '-1 m', '-1.23 m')), (True, -0.00123456789, ('−1.23457 m', '−1 m', '−1.23 m')), (True, -0.0, ('0', '0', '0.00')), (True, -0, ('0', '0', '0.00')), (True, 0, ('0', '0', '0.00')), (True, 1.23456789e-06, ('1.23457 µ', '1 µ', '1.23 µ')), (True, 0.123456789, ('123.457 m', '123 m', '123.46 m')), (True, 0.1, ('100 m', '100 m', '100.00 m')), (True, 1, ('1', '1', '1.00')), (True, 1.23456789, ('1.23457', '1', '1.23')), (True, 999.9, ('999.9', '1 k', '999.90')), (True, 999.9999, ('1 k', '1 k', '1.00 k')), (False, -999.9999, ('-1 k', '-1 k', '-1.00 k')), (True, -999.9999, ('−1 k', '−1 k', '−1.00 k')), (True, 1000, ('1 k', '1 k', '1.00 k')), (True, 1001, ('1.001 k', '1 k', '1.00 k')), (True, 100001, ('100.001 k', '100 k', '100.00 k')), (True, 987654.321, ('987.654 k', '988 k', '987.65 k')), (True, 1.23e+33, ('1230 Q', '1230 Q', '1230.00 Q'))]

    @pytest.mark.parametrize('unicode_minus, input, expected', raw_format_data)
    def test_params(self, unicode_minus, input, expected):
        """
        Test the formatting of EngFormatter for various values of the 'places'
        argument, in several cases:

        0. without a unit symbol but with a (default) space separator;
        1. with both a unit symbol and a (default) space separator;
        2. with both a unit symbol and some non default separators;
        3. without a unit symbol but with some non default separators.

        Note that cases 2. and 3. are looped over several separator strings.
        """
        plt.rcParams['axes.unicode_minus'] = unicode_minus
        UNIT = 's'
        DIGITS = '0123456789'
        exp_outputs = expected
        formatters = (mticker.EngFormatter(), mticker.EngFormatter(places=0), mticker.EngFormatter(places=2))
        for _formatter, _exp_output in zip(formatters, exp_outputs):
            assert _formatter(input) == _exp_output
        exp_outputs = (_s + ' ' + UNIT if _s[-1] in DIGITS else _s + UNIT for _s in expected)
        formatters = (mticker.EngFormatter(unit=UNIT), mticker.EngFormatter(unit=UNIT, places=0), mticker.EngFormatter(unit=UNIT, places=2))
        for _formatter, _exp_output in zip(formatters, exp_outputs):
            assert _formatter(input) == _exp_output
        for _sep in ('', '\u202f', '@_@'):
            exp_outputs = (_s + _sep + UNIT if _s[-1] in DIGITS else _s.replace(' ', _sep) + UNIT for _s in expected)
            formatters = (mticker.EngFormatter(unit=UNIT, sep=_sep), mticker.EngFormatter(unit=UNIT, places=0, sep=_sep), mticker.EngFormatter(unit=UNIT, places=2, sep=_sep))
            for _formatter, _exp_output in zip(formatters, exp_outputs):
                assert _formatter(input) == _exp_output
            exp_outputs = (_s.replace(' ', _sep) for _s in expected)
            formatters = (mticker.EngFormatter(sep=_sep), mticker.EngFormatter(places=0, sep=_sep), mticker.EngFormatter(places=2, sep=_sep))
            for _formatter, _exp_output in zip(formatters, exp_outputs):
                assert _formatter(input) == _exp_output