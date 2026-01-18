from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class PyRun(ExecutionControlCommandBase):
    """Run the program."""
    invoke = dont_suppress_errors(ExecutionControlCommandBase.run)