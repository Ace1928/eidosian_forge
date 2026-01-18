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
class BoxerWhq(BoxerIndexed):

    def __init__(self, discourse_id, sent_index, word_indices, ans_types, drs1, variable, drs2):
        BoxerIndexed.__init__(self, discourse_id, sent_index, word_indices)
        self.ans_types = ans_types
        self.drs1 = drs1
        self.variable = variable
        self.drs2 = drs2

    def _variables(self):
        return tuple(map(operator.or_, ({self.variable}, set(), set()), self.drs1._variables(), self.drs2._variables()))

    def atoms(self):
        return self.drs1.atoms() | self.drs2.atoms()

    def clean(self):
        return BoxerWhq(self.discourse_id, self.sent_index, self.word_indices, self.ans_types, self.drs1.clean(), self.variable, self.drs2.clean())

    def renumber_sentences(self, f):
        return BoxerWhq(self.discourse_id, f(self.sent_index), self.word_indices, self.ans_types, self.drs1, self.variable, self.drs2)

    def __iter__(self):
        return iter(('[' + ','.join(self.ans_types) + ']', self.drs1, self.variable, self.drs2))

    def _pred(self):
        return 'whq'