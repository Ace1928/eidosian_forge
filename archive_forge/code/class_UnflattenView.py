from __future__ import absolute_import, print_function, division
import itertools
import collections
import operator
from petl.compat import next, text_type
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, rowgetter, values, itervalues, \
from petl.transform.sorts import sort
class UnflattenView(Table):

    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            self.input = args[0]
            self.period = args[1]
        elif len(args) == 3:
            self.input = values(args[0], args[1])
            self.period = args[2]
        else:
            assert False, 'invalid arguments'
        self.missing = kwargs.get('missing', None)

    def __iter__(self):
        inpt = self.input
        period = self.period
        missing = self.missing
        outhdr = tuple(('f%s' % i for i in range(period)))
        yield outhdr
        row = list()
        for v in inpt:
            if len(row) < period:
                row.append(v)
            else:
                yield tuple(row)
                row = [v]
        if len(row) > 0:
            if len(row) < period:
                row.extend([missing] * (period - len(row)))
            yield tuple(row)