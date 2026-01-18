import re
import types
import sys
import os.path
import inspect
import base64
import warnings
class Production(object):
    reduced = 0

    def __init__(self, number, name, prod, precedence=('right', 0), func=None, file='', line=0):
        self.name = name
        self.prod = tuple(prod)
        self.number = number
        self.func = func
        self.callable = None
        self.file = file
        self.line = line
        self.prec = precedence
        self.len = len(self.prod)
        self.usyms = []
        for s in self.prod:
            if s not in self.usyms:
                self.usyms.append(s)
        self.lr_items = []
        self.lr_next = None
        if self.prod:
            self.str = '%s -> %s' % (self.name, ' '.join(self.prod))
        else:
            self.str = '%s -> <empty>' % self.name

    def __str__(self):
        return self.str

    def __repr__(self):
        return 'Production(' + str(self) + ')'

    def __len__(self):
        return len(self.prod)

    def __nonzero__(self):
        return 1

    def __getitem__(self, index):
        return self.prod[index]

    def lr_item(self, n):
        if n > len(self.prod):
            return None
        p = LRItem(self, n)
        try:
            p.lr_after = Prodnames[p.prod[n + 1]]
        except (IndexError, KeyError):
            p.lr_after = []
        try:
            p.lr_before = p.prod[n - 1]
        except IndexError:
            p.lr_before = None
        return p

    def bind(self, pdict):
        if self.func:
            self.callable = pdict[self.func]