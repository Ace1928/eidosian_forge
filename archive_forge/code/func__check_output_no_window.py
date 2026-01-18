import os
from os.path import join
import shutil
import tempfile
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex
def _check_output_no_window(*args, **kwargs):
    if os.name == 'nt':
        creation_flag = 134217728
    else:
        creation_flag = 0
    return check_output(*args, creationflags=creation_flag, **kwargs)