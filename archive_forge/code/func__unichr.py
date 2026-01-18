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
def _unichr(x):
    if x < 65536:
        return unichr(x)
    x -= 65536
    ch1 = 55296 | x >> 10
    ch2 = 56320 | x & 1023
    return unichr(ch1) + unichr(ch2)