from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_rightjoin(rightjoin_impl):
    _test_rightjoin_1(rightjoin_impl)
    _test_rightjoin_2(rightjoin_impl)
    _test_rightjoin_3(rightjoin_impl)
    _test_rightjoin_empty(rightjoin_impl)
    _test_rightjoin_novaluefield(rightjoin_impl)
    _test_rightjoin_prefix(rightjoin_impl)
    _test_rightjoin_lrkey(rightjoin_impl)
    _test_rightjoin_multiple(rightjoin_impl)