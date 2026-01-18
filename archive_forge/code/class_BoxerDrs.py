import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
class BoxerDrs(AbstractBoxerDrs):

    def __init__(self, refs, conds, consequent=None):
        AbstractBoxerDrs.__init__(self)
        self.refs = refs
        self.conds = conds
        self.consequent = consequent

    def _variables(self):
        variables = (set(), set(), set())
        for cond in self.conds:
            for s, v in zip(variables, cond._variables()):
                s.update(v)
        if self.consequent is not None:
            for s, v in zip(variables, self.consequent._variables()):
                s.update(v)
        return variables

    def atoms(self):
        atoms = reduce(operator.or_, (cond.atoms() for cond in self.conds), set())
        if self.consequent is not None:
            atoms.update(self.consequent.atoms())
        return atoms

    def clean(self):
        consequent = self.consequent.clean() if self.consequent else None
        return BoxerDrs(self.refs, [c.clean() for c in self.conds], consequent)

    def renumber_sentences(self, f):
        consequent = self.consequent.renumber_sentences(f) if self.consequent else None
        return BoxerDrs(self.refs, [c.renumber_sentences(f) for c in self.conds], consequent)

    def __repr__(self):
        s = 'drs([{}], [{}])'.format(', '.join(('%s' % r for r in self.refs)), ', '.join(('%s' % c for c in self.conds)))
        if self.consequent is not None:
            s = f'imp({s}, {self.consequent})'
        return s

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.refs == other.refs and (len(self.conds) == len(other.conds)) and reduce(operator.and_, (c1 == c2 for c1, c2 in zip(self.conds, other.conds))) and (self.consequent == other.consequent)

    def __ne__(self, other):
        return not self == other
    __hash__ = AbstractBoxerDrs.__hash__