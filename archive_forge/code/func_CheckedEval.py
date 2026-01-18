import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def CheckedEval(file_contents):
    """Return the eval of a gyp file.
  The gyp file is restricted to dictionaries and lists only, and
  repeated keys are not allowed.
  Note that this is slower than eval() is.
  """
    syntax_tree = ast.parse(file_contents)
    assert isinstance(syntax_tree, ast.Module)
    c1 = syntax_tree.body
    assert len(c1) == 1
    c2 = c1[0]
    assert isinstance(c2, ast.Expr)
    return CheckNode(c2.value, [])