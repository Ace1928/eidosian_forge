from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, itervalues
from petl.transform.sorts import sort

    Return True if there are no duplicate values for the given field(s),
    otherwise False. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b'],
        ...           ['b', 2],
        ...           ['c', 3, True]]
        >>> etl.isunique(table1, 'foo')
        False
        >>> etl.isunique(table1, 'bar')
        True

    The `field` argument can be a single field name or index (starting from
    zero) or a tuple of field names and/or indexes.

    