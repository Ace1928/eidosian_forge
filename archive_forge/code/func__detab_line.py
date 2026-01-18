import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _detab_line(self, line):
    """Recusively convert tabs to spaces in a single line.

        Called from _detab()."""
    if '\t' not in line:
        return line
    chunk1, chunk2 = line.split('\t', 1)
    chunk1 += ' ' * (self.tab_width - len(chunk1) % self.tab_width)
    output = chunk1 + chunk2
    return self._detab_line(output)