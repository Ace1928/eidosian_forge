import errno
import os
import subprocess as sp
import sys
from ._process_common import getoutput, arg_split
from IPython.utils.encoding import DEFAULT_ENCODING
Execute a command in a subshell.

        Parameters
        ----------
        cmd : str
            A command to be executed in the system shell.

        Returns
        -------
        int : child's exitstatus
        