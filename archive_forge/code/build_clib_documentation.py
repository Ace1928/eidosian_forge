import os
from glob import glob
import shutil
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError, DistutilsError, \
from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import (
from numpy.distutils.ccompiler_opt import new_ccompiler_opt
 Assemble flags from flag list

        Parameters
        ----------
        in_flags : None or sequence
            None corresponds to empty list.  Sequence elements can be strings
            or callables that return lists of strings. Callable takes `self` as
            single parameter.

        Returns
        -------
        out_flags : list
        