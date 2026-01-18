from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class ProxyException(Exception):

    def __init__(self, tp_name, args):
        self.tp_name = tp_name
        self.args = args

    def __repr__(self):
        return '%s%r' % (self.tp_name, self.args)