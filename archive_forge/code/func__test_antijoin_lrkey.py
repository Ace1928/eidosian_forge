from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_antijoin_lrkey(antijoin_impl):
    table1 = (('id', 'colour'), (0, 'black'), (1, 'blue'), (2, 'red'), (4, 'yellow'), (5, 'white'))
    table2 = (('identifier', 'shape'), (1, 'circle'), (3, 'square'))
    table3 = antijoin_impl(table1, table2, lkey='id', rkey='identifier')
    expect3 = (('id', 'colour'), (0, 'black'), (2, 'red'), (4, 'yellow'), (5, 'white'))
    ieq(expect3, table3)