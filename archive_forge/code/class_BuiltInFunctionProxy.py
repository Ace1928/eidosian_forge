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
class BuiltInFunctionProxy(object):

    def __init__(self, ml_name):
        self.ml_name = ml_name

    def __repr__(self):
        return '<built-in function %s>' % self.ml_name