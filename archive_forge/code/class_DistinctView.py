from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, itervalues
from petl.transform.sorts import sort
class DistinctView(Table):

    def __init__(self, table, key=None, count=None, presorted=False, buffersize=None, tempdir=None, cache=True):
        if presorted:
            self.table = table
        else:
            self.table = sort(table, key=key, buffersize=buffersize, tempdir=tempdir, cache=cache)
        self.key = key
        self.count = count

    def __iter__(self):
        it = iter(self.table)
        try:
            hdr = next(it)
        except StopIteration:
            return
        if self.key is None:
            indices = range(len(hdr))
        else:
            indices = asindices(hdr, self.key)
        getkey = operator.itemgetter(*indices)
        INIT = object()
        if self.count:
            hdr = tuple(hdr) + (self.count,)
            yield hdr
            previous = INIT
            n_dup = 1
            for row in it:
                if previous is INIT:
                    previous = row
                else:
                    kprev = getkey(previous)
                    kcurr = getkey(row)
                    if kprev == kcurr:
                        n_dup += 1
                    else:
                        yield (tuple(previous) + (n_dup,))
                        n_dup = 1
                        previous = row
            yield (tuple(previous) + (n_dup,))
        else:
            yield tuple(hdr)
            previous_keys = INIT
            for row in it:
                keys = getkey(row)
                if keys != previous_keys:
                    yield tuple(row)
                previous_keys = keys