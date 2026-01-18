from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_join_empty(join_impl):
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'), (3, 'purple'))
    table2 = (('id', 'shape'),)
    table3 = join_impl(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'),)
    ieq(expect3, table3)
    table1 = (('id', 'colour'),)
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'), (4, 'ellipse'))
    table3 = join_impl(table1, table2, key='id')
    expect3 = (('id', 'colour', 'shape'),)
    ieq(expect3, table3)