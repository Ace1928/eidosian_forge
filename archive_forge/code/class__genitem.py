import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
from warnings import warn
from six import advance_iterator, integer_types
from six.moves import _thread, range
from ._common import weekday as weekdaybase
class _genitem(object):

    def __init__(self, genlist, gen):
        try:
            self.dt = advance_iterator(gen)
            genlist.append(self)
        except StopIteration:
            pass
        self.genlist = genlist
        self.gen = gen

    def __next__(self):
        try:
            self.dt = advance_iterator(self.gen)
        except StopIteration:
            if self.genlist[0] is self:
                heapq.heappop(self.genlist)
            else:
                self.genlist.remove(self)
                heapq.heapify(self.genlist)
    next = __next__

    def __lt__(self, other):
        return self.dt < other.dt

    def __gt__(self, other):
        return self.dt > other.dt

    def __eq__(self, other):
        return self.dt == other.dt

    def __ne__(self, other):
        return self.dt != other.dt