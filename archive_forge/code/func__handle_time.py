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
def _handle_time(self, arg):
    conds = []
    self._parse_index_list()
    hour = self.token()
    if hour != 'XX':
        conds.append(self._make_atom('r_hour_2', arg, hour))
    self.assertToken(self.token(), ',')
    self._parse_index_list()
    min = self.token()
    if min != 'XX':
        conds.append(self._make_atom('r_min_2', arg, min))
    self.assertToken(self.token(), ',')
    self._parse_index_list()
    sec = self.token()
    if sec != 'XX':
        conds.append(self._make_atom('r_sec_2', arg, sec))
    return conds