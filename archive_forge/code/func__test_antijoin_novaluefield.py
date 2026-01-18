from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_antijoin_novaluefield(antijoin_impl):
    table1 = (('id', 'colour'), (0, 'black'), (1, 'blue'), (2, 'red'), (4, 'yellow'), (5, 'white'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'))
    expect = (('id', 'colour'), (0, 'black'), (2, 'red'), (4, 'yellow'), (5, 'white'))
    actual = antijoin_impl(table1, table2, key='id')
    ieq(expect, actual)
    actual = antijoin_impl(cut(table1, 'id'), table2, key='id')
    ieq(cut(expect, 'id'), actual)
    actual = antijoin_impl(table1, cut(table2, 'id'), key='id')
    ieq(expect, actual)
    actual = antijoin_impl(cut(table1, 'id'), cut(table2, 'id'), key='id')
    ieq(cut(expect, 'id'), actual)