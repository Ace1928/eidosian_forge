import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class VerboseCachingTargetsMatcher(object):

    def __init__(self, targets, required_match_count=None):
        self.targets = targets
        if required_match_count is None:
            required_match_count = len(targets)
        self.cache = {}
        self.required_match_count = required_match_count
        self._num_allowed_errors = len(targets) - required_match_count
        self.num_lookups = self.num_cached_true = self.num_cached_false = 0
        self.num_search_true = self.num_search_false = self.num_matches = 0

    def shift_targets(self):
        assert self._num_allowed_errors >= 0, (self.required_match_count, self._num_allowed_errors)
        if self._num_allowed_errors > 1:
            self.targets = self.targets[1:]
            self._num_allowed_errors = len(self.targets) - self.required_match_count

    def __getitem__(self, smarts, missing=object()):
        self.num_lookups += 1
        x = self.cache.get(smarts, missing)
        if x is not missing:
            if x:
                self.num_cached_true += 1
            else:
                self.num_cached_false += 1
            return x
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            raise AssertionError('Bad SMARTS: %r' % (smarts,))
        for i, target in enumerate(self.targets):
            if not MATCH(target, pat):
                self.num_search_false += 1
                self.num_matches += i + 1
                self.cache[smarts] = False
                N = len(self.targets)
                return False
        self.num_matches += i + 1
        self.num_search_true += 1
        self.cache[smarts] = True
        return True

    def report(self):
        (print >> sys.stderr, '%d tests of %d unique SMARTS, cache: %d True %d False, search: %d True %d False (%d substructure tests)' % (self.num_lookups, len(self.cache), self.num_cached_true, self.num_cached_false, self.num_search_true, self.num_search_false, self.num_matches))