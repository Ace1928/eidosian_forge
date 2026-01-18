import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def _get_frame_locals(self, frame):
    """ "
        Accessing f_local of current frame reset the namespace, so we want to avoid
        that or the following can happen

        ipdb> foo
        "old"
        ipdb> foo = "new"
        ipdb> foo
        "new"
        ipdb> where
        ipdb> foo
        "old"

        So if frame is self.current_frame we instead return self.curframe_locals

        """
    if frame is self.curframe:
        return self.curframe_locals
    else:
        return frame.f_locals