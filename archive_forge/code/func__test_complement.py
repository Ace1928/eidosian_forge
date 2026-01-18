from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def _test_complement(f):
    _test_complement_1(f)
    _test_complement_2(f)
    _test_complement_3(f)
    _test_complement_4(f)
    _test_complement_none(f)