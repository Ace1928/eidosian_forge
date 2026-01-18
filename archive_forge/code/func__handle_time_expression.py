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
def _handle_time_expression(self, arg):
    tok = self.token()
    self.assertToken(self.token(), '(')
    if tok == 'date':
        conds = self._handle_date(arg)
    elif tok == 'time':
        conds = self._handle_time(arg)
    else:
        return None
    self.assertToken(self.token(), ')')
    return [lambda sent_index, word_indices: BoxerPred(self.discourse_id, sent_index, word_indices, arg, tok, 'n', 0)] + [lambda sent_index, word_indices: cond for cond in conds]