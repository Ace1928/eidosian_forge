import os
import sys
import prettytable
from cliff import utils
from . import base
from cliff import columns
def _do_fit(fit_width):
    if os.name == 'nt':
        return fit_width
    return sys.stdout.isatty() or fit_width