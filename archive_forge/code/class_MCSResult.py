import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class MCSResult(object):

    def __init__(self, num_atoms, num_bonds, smarts, completed):
        self.num_atoms = num_atoms
        self.num_bonds = num_bonds
        self.smarts = smarts
        self.completed = completed

    def __nonzero__(self):
        return self.smarts is not None