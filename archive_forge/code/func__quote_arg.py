import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log
def _quote_arg(arg):
    """
    Quote the argument for safe use in a shell command line.
    """
    if '"' not in arg and ' ' in arg:
        return '"%s"' % arg
    return arg