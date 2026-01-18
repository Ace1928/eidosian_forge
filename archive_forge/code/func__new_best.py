import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def _new_best(self, num_atoms, num_bonds, smarts):
    self.best_num_atoms = num_atoms
    self.best_num_bonds = num_bonds
    self.best_smarts = smarts
    self.sizes = sizes = (num_atoms, num_bonds)
    self.timer.mark('new best')
    if self.verbose:
        dt = self.timer.mark_times['new best'] - self.timer.mark_times['start fmcs']
        sys.stderr.write('Best after %.1fs: %d atoms %d bonds %s\n' % (dt, num_atoms, num_bonds, smarts))
    return sizes