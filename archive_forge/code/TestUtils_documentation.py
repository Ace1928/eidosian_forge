from __future__ import absolute_import
import os
import re
import unittest
import shlex
import sys
import tempfile
import textwrap
from io import open
from functools import partial
from .Compiler import Errors
from .CodeWriter import CodeWriter
from .Compiler.TreeFragment import TreeFragment, strip_common_indent
from .Compiler.Visitor import TreeVisitor, VisitorTransform
from .Compiler import TreePath
Calls func and succeeds if and only if no exception is raised
        (i.e. converts exception raising into a failed testcase). Returns
        the return value of func.