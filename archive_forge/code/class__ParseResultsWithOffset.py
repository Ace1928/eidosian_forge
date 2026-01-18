import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class _ParseResultsWithOffset(object):

    def __init__(self, p1, p2):
        self.tup = (p1, p2)

    def __getitem__(self, i):
        return self.tup[i]

    def __repr__(self):
        return repr(self.tup)

    def setOffset(self, i):
        self.tup = (self.tup[0], i)