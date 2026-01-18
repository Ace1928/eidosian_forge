from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def _add_line_info(self, msg, pos):
    msg = '%s at line %s column %s' % (msg, pos[0], pos[1])
    if self.name:
        msg += ' in file %s' % self.name
    return msg