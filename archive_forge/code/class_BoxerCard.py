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
class BoxerCard(BoxerIndexed):

    def __init__(self, discourse_id, sent_index, word_indices, var, value, type):
        BoxerIndexed.__init__(self, discourse_id, sent_index, word_indices)
        self.var = var
        self.value = value
        self.type = type

    def _variables(self):
        return ({self.var}, set(), set())

    def renumber_sentences(self, f):
        return BoxerCard(self.discourse_id, f(self.sent_index), self.word_indices, self.var, self.value, self.type)

    def __iter__(self):
        return iter((self.var, self.value, self.type))

    def _pred(self):
        return 'card'