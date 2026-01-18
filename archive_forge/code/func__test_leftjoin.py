from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_leftjoin(leftjoin_impl):
    _test_leftjoin_1(leftjoin_impl)
    _test_leftjoin_2(leftjoin_impl)
    _test_leftjoin_3(leftjoin_impl)
    _test_leftjoin_compound_keys(leftjoin_impl)
    _test_leftjoin_empty(leftjoin_impl)
    _test_leftjoin_novaluefield(leftjoin_impl)
    _test_leftjoin_multiple(leftjoin_impl)
    _test_leftjoin_prefix(leftjoin_impl)
    _test_leftjoin_lrkey(leftjoin_impl)