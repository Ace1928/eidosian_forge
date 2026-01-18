from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def _test_intersection_4(intersection_impl):
    table1 = (('foo', 'bar'), ('A', 1), ('B', 2), ('B', 2), ('B', 2), ('C', 7))
    table2 = (('foo', 'bar'), ('A', 9), ('B', 2), ('B', 2), ('B', 3))
    expectation = (('foo', 'bar'), ('B', 2), ('B', 2))
    result = intersection_impl(table1, table2)
    ieq(expectation, result)
    ieq(expectation, result)