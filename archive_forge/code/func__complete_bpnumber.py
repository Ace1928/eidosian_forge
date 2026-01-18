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
def _complete_bpnumber(self, text, line, begidx, endidx):
    return [str(i) for i, bp in enumerate(bdb.Breakpoint.bpbynumber) if bp is not None and str(i).startswith(text)]