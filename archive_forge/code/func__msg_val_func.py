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
def _msg_val_func(self, arg, func):
    try:
        val = self._getval(arg)
    except:
        return
    try:
        self.message(func(val))
    except:
        self._error_exc()