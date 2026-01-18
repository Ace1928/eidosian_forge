import re
import types
import sys
import os.path
import inspect
import base64
import warnings
class LRItem(object):

    def __init__(self, p, n):
        self.name = p.name
        self.prod = list(p.prod)
        self.number = p.number
        self.lr_index = n
        self.lookaheads = {}
        self.prod.insert(n, '.')
        self.prod = tuple(self.prod)
        self.len = len(self.prod)
        self.usyms = p.usyms

    def __str__(self):
        if self.prod:
            s = '%s -> %s' % (self.name, ' '.join(self.prod))
        else:
            s = '%s -> <empty>' % self.name
        return s

    def __repr__(self):
        return 'LRItem(' + str(self) + ')'