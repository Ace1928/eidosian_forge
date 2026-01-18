import sys
import os
import pprint
import re
from pathlib import Path
from itertools import dropwhile
import argparse
import copy
from . import crackfortran
from . import rules
from . import cb_rules
from . import auxfuncs
from . import cfuncs
from . import f90mod_rules
from . import __version__
from . import capi_maps
from numpy.f2py._backends import f2py_build_generator
class CombineIncludePaths(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        include_paths_set = set(getattr(namespace, 'include_paths', []) or [])
        if option_string == '--include_paths':
            outmess('Use --include-paths or -I instead of --include_paths which will be removed')
        if option_string == '--include-paths' or option_string == '--include_paths':
            include_paths_set.update(values.split(':'))
        else:
            include_paths_set.add(values)
        setattr(namespace, 'include_paths', list(include_paths_set))