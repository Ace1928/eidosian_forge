import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
@contextlib.contextmanager
def _type_param_scope(self, node):
    with contextlib.ExitStack() as ctx:
        if sys.version_info >= (3, 12):
            ctx.enter_context(self.in_scope(TypeScope))
            for param in node.type_params:
                self.handleNode(param, node)
        yield