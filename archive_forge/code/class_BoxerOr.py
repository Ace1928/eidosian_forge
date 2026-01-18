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
class BoxerOr(BoxerIndexed):

    def __init__(self, discourse_id, sent_index, word_indices, drs1, drs2):
        BoxerIndexed.__init__(self, discourse_id, sent_index, word_indices)
        self.drs1 = drs1
        self.drs2 = drs2

    def _variables(self):
        return tuple(map(operator.or_, self.drs1._variables(), self.drs2._variables()))

    def atoms(self):
        return self.drs1.atoms() | self.drs2.atoms()

    def clean(self):
        return BoxerOr(self.discourse_id, self.sent_index, self.word_indices, self.drs1.clean(), self.drs2.clean())

    def renumber_sentences(self, f):
        return BoxerOr(self.discourse_id, f(self.sent_index), self.word_indices, self.drs1, self.drs2)

    def __iter__(self):
        return iter((self.drs1, self.drs2))

    def _pred(self):
        return 'or'