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
def CLASSDEF(self, node):
    """
        Check names used in a class definition, including its decorators, base
        classes, and the body of its definition.  Additionally, add its name to
        the current scope.
        """
    for deco in node.decorator_list:
        self.handleNode(deco, node)
    with self._type_param_scope(node):
        for baseNode in node.bases:
            self.handleNode(baseNode, node)
        for keywordNode in node.keywords:
            self.handleNode(keywordNode, node)
        with self.in_scope(ClassScope):
            if self.withDoctest and (not self._in_doctest()) and (not isinstance(self.scope, FunctionScope)):
                self.deferFunction(lambda: self.handleDoctests(node))
            for stmt in node.body:
                self.handleNode(stmt, node)
    self.addBinding(node, ClassDefinition(node.name, node))