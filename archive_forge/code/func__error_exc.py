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
def _error_exc(self):
    exc_info = sys.exc_info()[:2]
    self.error(traceback.format_exception_only(*exc_info)[-1].strip())