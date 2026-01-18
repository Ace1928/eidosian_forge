from __future__ import print_function, absolute_import, division, unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
class TestCalculations(object):

    def test_mul_00(self):
        d = round_trip_load('        - 0.1\n        ')
        d[0] *= -1
        x = round_trip_dump(d)
        assert x == '- -0.1\n'