from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def __use_fuzzy_mode(self, label):
    """@see: L{split_label}"""
    return self.split_label_fuzzy(label)