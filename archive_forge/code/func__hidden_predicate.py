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
def _hidden_predicate(self, frame):
    """
        Given a frame return whether it it should be hidden or not by IPython.
        """
    if self._predicates['readonly']:
        fname = frame.f_code.co_filename
        if os.path.isfile(fname) and (not os.access(fname, os.W_OK)):
            return True
    if self._predicates['tbhide']:
        if frame in (self.curframe, getattr(self, 'initial_frame', None)):
            return False
        frame_locals = self._get_frame_locals(frame)
        if '__tracebackhide__' not in frame_locals:
            return False
        return frame_locals['__tracebackhide__']
    return False