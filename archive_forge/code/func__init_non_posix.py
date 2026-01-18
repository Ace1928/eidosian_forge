import os
import sys
from os.path import pardir, realpath
def _init_non_posix(vars):
    """Initialize the module as appropriate for NT"""
    import _imp
    vars['LIBDEST'] = get_path('stdlib')
    vars['BINLIBDEST'] = get_path('platstdlib')
    vars['INCLUDEPY'] = get_path('include')
    vars['EXT_SUFFIX'] = _imp.extension_suffixes()[0]
    vars['EXE'] = '.exe'
    vars['VERSION'] = _PY_VERSION_SHORT_NO_DOT
    vars['BINDIR'] = os.path.dirname(_safe_realpath(sys.executable))
    vars['TZPATH'] = ''