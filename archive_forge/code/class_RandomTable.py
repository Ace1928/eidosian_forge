from __future__ import absolute_import, print_function, division
import hashlib
import random as pyrandom
import time
from collections import OrderedDict
from functools import partial
from petl.compat import xrange, text_type
from petl.util.base import Table
class RandomTable(Table):

    def __init__(self, numflds=5, numrows=100, wait=0, seed=None):
        self.numflds = numflds
        self.numrows = numrows
        self.wait = wait
        if seed is None:
            self.seed = randomseed()
        else:
            self.seed = seed

    def __iter__(self):
        nf = self.numflds
        nr = self.numrows
        seed = self.seed
        pyrandom.seed(seed)
        flds = ['f%s' % n for n in range(nf)]
        yield tuple(flds)
        for _ in xrange(nr):
            if self.wait:
                time.sleep(self.wait)
            yield tuple((pyrandom.random() for n in range(nf)))

    def reseed(self):
        self.seed = randomseed()