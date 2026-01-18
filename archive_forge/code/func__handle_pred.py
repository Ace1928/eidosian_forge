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
def _handle_pred(self):
    self.assertToken(self.token(), '(')
    variable = self.parse_variable()
    self.assertToken(self.token(), ',')
    name = self.token()
    self.assertToken(self.token(), ',')
    pos = self.token()
    self.assertToken(self.token(), ',')
    sense = int(self.token())
    self.assertToken(self.token(), ')')

    def _handle_pred_f(sent_index, word_indices):
        return BoxerPred(self.discourse_id, sent_index, word_indices, variable, name, pos, sense)
    return _handle_pred_f