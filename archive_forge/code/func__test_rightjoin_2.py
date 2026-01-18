from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_rightjoin_2(rightjoin_impl):
    table1 = (('id', 'colour'), (0, 'black'), (1, 'blue'), (2, 'red'), (3, 'purple'), (5, 'yellow'), (7, 'white'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'), (4, 'ellipse'))
    table3 = rightjoin_impl(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'), (1, 'blue', 'circle'), (3, 'purple', 'square'), (4, None, 'ellipse'))
    ieq(expect3, table3)
    ieq(expect3, table3)
    table4 = rightjoin_impl(table1, table2)
    expect4 = expect3
    ieq(expect4, table4)