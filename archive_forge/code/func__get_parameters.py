from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def _get_parameters(function):
    if sys.version_info >= (3, 3):
        return [parameter.name for parameter in inspect.signature(function).parameters.values() if parameter.kind == parameter.POSITIONAL_OR_KEYWORD]
    else:
        return inspect.getargspec(function)[0]