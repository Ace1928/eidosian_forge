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
def _handle_alfa(self, make_callback):
    self.assertToken(self.token(), '(')
    type = self.token()
    self.assertToken(self.token(), ',')
    drs1 = self.process_next_expression(None)
    self.assertToken(self.token(), ',')
    drs2 = self.process_next_expression(None)
    self.assertToken(self.token(), ')')
    return lambda sent_index, word_indices: make_callback(sent_index, word_indices, drs1, drs2)