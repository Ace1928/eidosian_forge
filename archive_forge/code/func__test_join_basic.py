from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_join_basic(join_impl):
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'), (4, 'ellipse'))
    table3 = join_impl(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'), (1, 'blue', 'circle'), (3, 'purple', 'square'))
    ieq(expect3, table3)
    ieq(expect3, table3)
    table4 = join_impl(table1, table2)
    expect4 = expect3
    ieq(expect4, table4)
    ieq(expect4, table4)
    table5 = (('id', 'colour'), (1, 'blue'), (1, 'red'), (2, 'purple'))
    table6 = (('id', 'shape'), (1, 'circle'), (1, 'square'), (2, 'ellipse'))
    table7 = join_impl(table5, table6, key='id')
    expect7 = (('id', 'colour', 'shape'), (1, 'blue', 'circle'), (1, 'blue', 'square'), (1, 'red', 'circle'), (1, 'red', 'square'), (2, 'purple', 'ellipse'))
    ieq(expect7, table7)