from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_rightjoin_3(rightjoin_impl):
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'), (4, 'orange'))
    table2 = (('id', 'shape'), (0, 'triangle'), (1, 'circle'), (3, 'square'), (5, 'ellipse'), (7, 'pentagon'))
    table3 = rightjoin_impl(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'), (0, None, 'triangle'), (1, 'blue', 'circle'), (3, 'purple', 'square'), (5, None, 'ellipse'), (7, None, 'pentagon'))
    ieq(expect3, table3)
    ieq(expect3, table3)
    table4 = rightjoin_impl(table1, table2)
    expect4 = expect3
    ieq(expect4, table4)