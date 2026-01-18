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
def _sent_and_word_indices(self, indices):
    """
        :return: list of (sent_index, word_indices) tuples
        """
    sent_indices = {i / 1000 - 1 for i in indices if i >= 0}
    if sent_indices:
        pairs = []
        for sent_index in sent_indices:
            word_indices = [i % 1000 - 1 for i in indices if sent_index == i / 1000 - 1]
            pairs.append((sent_index, word_indices))
        return pairs
    else:
        word_indices = [i % 1000 - 1 for i in indices]
        return [(None, word_indices)]