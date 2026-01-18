import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class VerboseHeapOps(object):

    def __init__(self, trigger, verboseDelay):
        self.num_seeds_added = 0
        self.num_seeds_processed = 0
        self.verboseDelay = verboseDelay
        self._time_for_next_report = time.perf_counter() + verboseDelay
        self.trigger = trigger

    def heappush(self, seeds, item):
        self.num_seeds_added += 1
        return heappush(seeds, item)

    def heappop(self, seeds):
        if time.perf_counter() >= self._time_for_next_report:
            self.trigger()
            self.report()
            self._time_for_next_report = time.perf_counter() + self.verboseDelay
        self.num_seeds_processed += 1
        return heappop(seeds)

    def trigger_report(self):
        self.trigger()
        self.report()

    def report(self):
        (print >> sys.stderr, '  %d subgraphs enumerated, %d processed' % (self.num_seeds_added, self.num_seeds_processed))