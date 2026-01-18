import os
import sys
from types import ModuleType
from .version import version as __version__  # NOQA:F401
def _py_abspath(path):
    """
    special version of abspath
    that will leave paths from jython jars alone
    """
    if path.startswith('__pyclasspath__'):
        return path
    else:
        return os.path.abspath(path)