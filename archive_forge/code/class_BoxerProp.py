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
class BoxerProp(BoxerIndexed):

    def __init__(self, discourse_id, sent_index, word_indices, var, drs):
        BoxerIndexed.__init__(self, discourse_id, sent_index, word_indices)
        self.var = var
        self.drs = drs

    def _variables(self):
        return tuple(map(operator.or_, (set(), set(), {self.var}), self.drs._variables()))

    def referenced_labels(self):
        return {self.drs}

    def atoms(self):
        return self.drs.atoms()

    def clean(self):
        return BoxerProp(self.discourse_id, self.sent_index, self.word_indices, self.var, self.drs.clean())

    def renumber_sentences(self, f):
        return BoxerProp(self.discourse_id, f(self.sent_index), self.word_indices, self.var, self.drs.renumber_sentences(f))

    def __iter__(self):
        return iter((self.var, self.drs))

    def _pred(self):
        return 'prop'