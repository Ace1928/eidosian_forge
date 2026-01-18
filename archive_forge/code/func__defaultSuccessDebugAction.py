import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _defaultSuccessDebugAction(instring, startloc, endloc, expr, toks):
    print('Matched ' + _ustr(expr) + ' -> ' + str(toks.asList()))