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
def _parse_index_list(self):
    indices = []
    self.assertToken(self.token(), '[')
    while self.token(0) != ']':
        indices.append(self.parse_index())
        if self.token(0) == ',':
            self.token()
    self.token()
    self.assertToken(self.token(), ':')
    return indices