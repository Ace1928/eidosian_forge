import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log

    Quote the argument for safe use in a shell command line.
    