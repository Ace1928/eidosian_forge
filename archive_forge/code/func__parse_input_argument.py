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
def _parse_input_argument(arg: str) -> typing.Tuple[str, str]:
    """Process the input to an R magic commmand (`%R`, `%%R`, `%Rpush`)."""
    arg_elts = arg.split('=', maxsplit=1)
    if len(arg_elts) == 1:
        rhs = arg_elts[0]
        lhs = rhs
    else:
        lhs, rhs = arg_elts
    return (lhs, rhs)