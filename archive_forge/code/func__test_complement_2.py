from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def _test_complement_2(complement_impl):
    tablea = (('foo', 'bar', 'baz'), ('A', 1, True), ('C', 7, False), ('B', 2, False), ('C', 9, True))
    tableb = (('x', 'y', 'z'), ('B', 2, False), ('A', 9, False), ('B', 3, True), ('C', 9, True))
    aminusb = (('foo', 'bar', 'baz'), ('A', 1, True), ('C', 7, False))
    result = complement_impl(tablea, tableb)
    ieq(aminusb, result)
    bminusa = (('x', 'y', 'z'), ('A', 9, False), ('B', 3, True))
    result = complement_impl(tableb, tablea)
    ieq(bminusa, result)