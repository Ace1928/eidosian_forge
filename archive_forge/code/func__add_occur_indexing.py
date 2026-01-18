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
def _add_occur_indexing(self, base, ex):
    if self._occur_index and ex.sent_index is not None:
        if ex.discourse_id:
            base += '_%s' % ex.discourse_id
        base += '_s%s' % ex.sent_index
        base += '_w%s' % sorted(ex.word_indices)[0]
    return base