import os
import sys
from os.path import pardir, realpath
def _init_posix(vars):
    """Initialize the module as appropriate for POSIX systems."""
    name = _get_sysconfigdata_name()
    _temp = __import__(name, globals(), locals(), ['build_time_vars'], 0)
    build_time_vars = _temp.build_time_vars
    vars.update(build_time_vars)