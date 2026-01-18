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
@classmethod
def _dummy_key(self):
    return gdb.lookup_global_symbol('_PySet_Dummy').value()