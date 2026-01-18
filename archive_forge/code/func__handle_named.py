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
def _handle_named(self):
    self.assertToken(self.token(), '(')
    variable = self.parse_variable()
    self.assertToken(self.token(), ',')
    name = self.token()
    self.assertToken(self.token(), ',')
    type = self.token()
    self.assertToken(self.token(), ',')
    sense = self.token()
    self.assertToken(self.token(), ')')
    return lambda sent_index, word_indices: BoxerNamed(self.discourse_id, sent_index, word_indices, variable, name, type, sense)