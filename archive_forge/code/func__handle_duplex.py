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
def _handle_duplex(self):
    self.assertToken(self.token(), '(')
    ans_types = []
    self.assertToken(self.token(), 'whq')
    self.assertToken(self.token(), ',')
    d1 = self.process_next_expression(None)
    self.assertToken(self.token(), ',')
    ref = self.parse_variable()
    self.assertToken(self.token(), ',')
    d2 = self.process_next_expression(None)
    self.assertToken(self.token(), ')')
    return lambda sent_index, word_indices: BoxerWhq(self.discourse_id, sent_index, word_indices, ans_types, d1, ref, d2)