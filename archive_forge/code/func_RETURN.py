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
def RETURN(self, node):
    if isinstance(self.scope, (ClassScope, ModuleScope)):
        self.report(messages.ReturnOutsideFunction, node)
        return
    if node.value and hasattr(self.scope, 'returnValue') and (not self.scope.returnValue):
        self.scope.returnValue = node.value
    self.handleNode(node.value, node)