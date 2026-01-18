from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_leftjoin_empty(leftjoin_impl):
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'), (5, 'yellow'), (7, 'orange'))
    table2 = (('id', 'shape'),)
    table3 = leftjoin_impl(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'), (1, 'blue', None), (2, 'red', None), (3, 'purple', None), (5, 'yellow', None), (7, 'orange', None))
    ieq(expect3, table3)