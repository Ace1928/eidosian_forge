import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class CachingTargetsMatcher(dict):

    def __init__(self, targets, required_match_count=None):
        self.targets = targets
        if required_match_count is None:
            required_match_count = len(targets)
        self.required_match_count = required_match_count
        self._num_allowed_errors = len(targets) - required_match_count
        super(dict, self).__init__()

    def shift_targets(self):
        assert self._num_allowed_errors >= 0, (self.required_match_count, self._num_allowed_errors)
        self.targets = self.targets[1:]
        self._num_allowed_errors = len(self.targets) - self.required_match_count

    def __missing__(self, smarts):
        num_allowed_errors = self._num_allowed_errors
        if num_allowed_errors < 0:
            raise AssertionError('I should never be called')
            self[smarts] = False
            return False
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            raise AssertionError('Bad SMARTS: %r' % (smarts,))
        num_allowed_errors = self._num_allowed_errors
        for target in self.targets:
            if not MATCH(target, pat):
                if num_allowed_errors == 0:
                    self[smarts] = False
                    return False
                num_allowed_errors -= 1
        self[smarts] = True
        return True