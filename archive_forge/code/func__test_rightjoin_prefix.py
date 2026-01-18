from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_rightjoin_prefix(rightjoin_impl):
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'))
    table2 = (('id', 'shape'), (0, 'triangle'), (1, 'circle'), (3, 'square'), (4, 'ellipse'), (5, 'pentagon'))
    table3 = rightjoin_impl(table1, table2, key='id', lprefix='l_', rprefix='r_')
    expect3 = (('l_id', 'l_colour', 'r_shape'), (0, None, 'triangle'), (1, 'blue', 'circle'), (3, 'purple', 'square'), (4, None, 'ellipse'), (5, None, 'pentagon'))
    ieq(expect3, table3)