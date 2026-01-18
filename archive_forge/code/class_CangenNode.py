import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class CangenNode(object):
    __slots__ = ['index', 'atom_smarts', 'value', 'neighbors', 'rank', 'outgoing_edges']

    def __init__(self, index, atom_smarts):
        self.index = index
        self.atom_smarts = atom_smarts
        self.value = 0
        self.neighbors = []
        self.rank = 0
        self.outgoing_edges = []