import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log
def _update_environment(**env):
    log.debug('_update_environment(...)')
    for name, value in env.items():
        os.environ[name] = value or ''