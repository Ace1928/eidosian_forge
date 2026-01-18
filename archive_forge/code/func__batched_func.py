from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def _batched_func(self, arg):
    if type(arg) is list:
        return [_func(self, n) for n in arg]
    else:
        return _func(self, arg)