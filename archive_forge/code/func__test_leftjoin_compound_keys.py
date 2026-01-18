from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_leftjoin_compound_keys(leftjoin_impl):
    table5 = (('id', 'time', 'height'), (1, 1, 12.3), (1, 2, 34.5), (2, 1, 56.7))
    table6 = (('id', 'time', 'weight', 'bp'), (1, 2, 4.5, 120), (2, 1, 6.7, 110), (2, 2, 8.9, 100))
    table7 = leftjoin_impl(table5, table6, key=['id', 'time'])
    expect7 = (('id', 'time', 'height', 'weight', 'bp'), (1, 1, 12.3, None, None), (1, 2, 34.5, 4.5, 120), (2, 1, 56.7, 6.7, 110))
    ieq(expect7, table7)