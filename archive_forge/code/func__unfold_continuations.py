import ast
import inspect
import io
import linecache
import re
import sys
import textwrap
import tokenize
import astunparse
import gast
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.util import tf_inspect
def _unfold_continuations(code_string):
    """Removes any backslash line continuations from the code."""
    return code_string.replace('\\\n', '')