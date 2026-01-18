from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def _test_intersection(intersection_impl):
    _test_intersection_1(intersection_impl)
    _test_intersection_2(intersection_impl)
    _test_intersection_3(intersection_impl)
    _test_intersection_4(intersection_impl)
    _test_intersection_empty(intersection_impl)