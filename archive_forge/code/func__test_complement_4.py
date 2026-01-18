from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def _test_complement_4(complement_impl):
    table1 = (('foo', 'bar'), ('A', 1), ('B', 2), ('B', 2), ('C', 7))
    table2 = (('foo', 'bar'), ('B', 2))
    result = complement_impl(table1, table2)
    expectation = (('foo', 'bar'), ('A', 1), ('B', 2), ('C', 7))
    ieq(expectation, result)
    ieq(expectation, result)
    result = complement_impl(table1, table2, strict=True)
    expectation = (('foo', 'bar'), ('A', 1), ('C', 7))
    ieq(expectation, result)
    ieq(expectation, result)