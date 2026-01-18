import contextlib
import sys
import tempfile
from glob import glob
import os
from shutil import rmtree
import textwrap
import typing
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface as ri
import rpy2.rinterface_lib.openrlib
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects.lib import grdevices
from rpy2.robjects.conversion import (Converter,
import warnings
import IPython.display  # type: ignore
from IPython.core import displaypub  # type: ignore
from IPython.core.magic import (Magics,   # type: ignore
from IPython.core.magic_arguments import (argument,  # type: ignore
def _import_name_into_r(self, arg: str, env: ri.SexpEnvironment, local_ns: dict) -> None:
    lhs, rhs = _parse_input_argument(arg)
    val = None
    try:
        val = _find(rhs, local_ns)
    except NameError:
        if self.shell is None:
            warnings.warn(f'The shell is None. Unable to look for {rhs}.')
        else:
            val = _find(rhs, self.shell.user_ns)
    if val is not None:
        env[lhs] = val