from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def _test_join_lrkey(join_impl):
    table1 = (('id', 'colour'), ('aa', 'blue'), ('bb', 'red'), ('cc', 'purple'))
    table2 = (('identifier', 'shape'), ('aa', 'circle'), ('cc', 'square'), ('dd', 'ellipse'))
    table3 = join_impl(table1, table2, lkey='id', rkey='identifier')
    expect3 = (('id', 'colour', 'shape'), ('aa', 'blue', 'circle'), ('cc', 'purple', 'square'))
    ieq(expect3, table3)