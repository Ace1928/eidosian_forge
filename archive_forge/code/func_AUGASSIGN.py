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
def AUGASSIGN(self, node):
    self.handleNodeLoad(node.target, node)
    self.handleNode(node.value, node)
    self.handleNode(node.target, node)