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
def _getval(self, arg):
    try:
        return eval(arg, self.curframe.f_globals, self.curframe_locals)
    except:
        self._error_exc()
        raise