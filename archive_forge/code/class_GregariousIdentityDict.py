from fontTools.misc.timeTools import timestampNow
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from functools import reduce
import operator
import logging
class GregariousIdentityDict(object):
    """A dictionary-like object that welcomes guests without reservations and
    adds them to the end of the guest list."""

    def __init__(self, lst):
        self.l = lst
        self.s = set((id(v) for v in lst))

    def __getitem__(self, v):
        if id(v) not in self.s:
            self.s.add(id(v))
            self.l.append(v)
        return v