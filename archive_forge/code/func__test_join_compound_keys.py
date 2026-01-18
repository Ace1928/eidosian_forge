from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_join_compound_keys(join_impl):
    table8 = (('id', 'time', 'height'), (1, 1, 12.3), (1, 2, 34.5), (2, 1, 56.7))
    table9 = (('id', 'time', 'weight'), (1, 2, 4.5), (2, 1, 6.7), (2, 2, 8.9))
    table10 = join_impl(table8, table9, key=['id', 'time'])
    expect10 = (('id', 'time', 'height', 'weight'), (1, 2, 34.5, 4.5), (2, 1, 56.7, 6.7))
    ieq(expect10, table10)
    table11 = join_impl(table8, table9)
    expect11 = expect10
    ieq(expect11, table11)