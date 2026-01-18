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
def EXCEPTHANDLER(self, node):
    if node.name is None:
        self.handleChildren(node)
        return
    if node.name in self.scope:
        self.handleNodeStore(node)
    try:
        prev_definition = self.scope.pop(node.name)
    except KeyError:
        prev_definition = None
    self.handleNodeStore(node)
    self.handleChildren(node)
    try:
        binding = self.scope.pop(node.name)
    except KeyError:
        pass
    else:
        if not binding.used:
            self.report(messages.UnusedVariable, node, node.name)
    if prev_definition:
        self.scope[node.name] = prev_definition