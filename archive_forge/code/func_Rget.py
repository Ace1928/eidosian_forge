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
@argument('output', nargs=1, type=str)
@line_magic
def Rget(self, line):
    """
        Return an object from rpy2, possibly as a structured array (if
        possible).
        Similar to Rpull except only one argument is accepted and the value is
        returned rather than pushed to self.shell.user_ns::

            In [3]: dtype=[('x', '<i4'), ('y', '<f8'), ('z', '|S1')]

            In [4]: datapy = np.array([(1, 2.9, 'a'), (2, 3.5, 'b'),
            ...                        (3, 2.1, 'c'), (4, 5, 'e')],
            ...                        dtype=dtype)

            In [5]: %R -i datapy

            In [6]: %Rget datapy
            Out[6]:
            array([['1', '2', '3', '4'],
                   ['2', '3', '2', '5'],
                   ['a', 'b', 'c', 'e']],
                  dtype='|S1')
        """
    args = parse_argstring(self.Rget, line)
    output = args.output
    with localconverter(self.converter):
        res = ro.globalenv.find(output[0])
    return res