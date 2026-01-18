import builtins
import inspect
import io
import keyword
import linecache
import os
import re
import sys
import sysconfig
import tokenize
import traceback
@staticmethod
def _get_lib_dirs():
    schemes = sysconfig.get_scheme_names()
    names = ['stdlib', 'platstdlib', 'platlib', 'purelib']
    paths = {sysconfig.get_path(name, scheme) for scheme in schemes for name in names}
    return [os.path.abspath(path).lower() + os.sep for path in paths if path in sys.path]