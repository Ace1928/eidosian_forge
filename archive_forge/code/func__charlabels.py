from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def _charlabels(self, options):
    """Get labels for characters (PRIVATE)."""
    self.charlabels = {}
    opts = CharBuffer(options)
    while True:
        w = opts.next_word()
        if w is None:
            break
        identifier = self._resolve(w, set_type=CHARSET)
        state = quotestrip(opts.next_word())
        self.charlabels[identifier] = state
        c = opts.next_nonwhitespace()
        if c is None:
            break
        elif c != ',':
            raise NexusError(f"Missing ',' in line {options}.")