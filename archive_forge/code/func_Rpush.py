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
@magic_arguments()
@argument('-c', '--converter', default=None, help=textwrap.dedent("\n        Name of local converter to use. A converter contains the rules to\n        convert objects back and forth between Python and R. If not\n        specified/None, the defaut converter for the magic's module is used\n        (that is rpy2's default converter + numpy converter + pandas converter\n        if all three are available)."))
@argument('inputs', nargs='*')
@needs_local_scope
@line_magic
def Rpush(self, line, local_ns=None):
    """
        A line-level magic that pushes
        variables from python to R. The line should be made up
        of whitespace separated variable names in the IPython
        namespace::

            In [7]: import numpy as np

            In [8]: X = np.array([4.5,6.3,7.9])

            In [9]: X.mean()
            Out[9]: 6.2333333333333343

            In [10]: %Rpush X

            In [11]: %R mean(X)
            Out[11]: array([ 6.23333333])

        """
    args = parse_argstring(self.Rpush, line)
    converter = self._find_converter(args.converter, local_ns)
    if local_ns is None:
        local_ns = {}
    with localconverter(converter):
        for arg in args.inputs:
            self._import_name_into_r(arg, ro.globalenv, local_ns)