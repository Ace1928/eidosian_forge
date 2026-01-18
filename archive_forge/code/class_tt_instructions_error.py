from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging
class tt_instructions_error(Exception):

    def __init__(self, error):
        self.error = error

    def __str__(self):
        return 'TT instructions error: %s' % repr(self.error)