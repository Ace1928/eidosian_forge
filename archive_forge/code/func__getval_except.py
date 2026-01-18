import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def _getval_except(self, arg, frame=None):
    try:
        if frame is None:
            return eval(arg, self.curframe.f_globals, self.curframe_locals)
        else:
            return eval(arg, frame.f_globals, frame.f_locals)
    except:
        exc_info = sys.exc_info()[:2]
        err = traceback.format_exception_only(*exc_info)[-1].strip()
        return _rstr('** raised %s **' % err)