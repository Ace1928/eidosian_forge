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
def IMPORTFROM(self, node):
    if node.module == '__future__':
        if not self.futuresAllowed:
            self.report(messages.LateFutureImport, node)
    else:
        self.futuresAllowed = False
    module = '.' * node.level + (node.module or '')
    for alias in node.names:
        name = alias.asname or alias.name
        if node.module == '__future__':
            importation = FutureImportation(name, node, self.scope)
            if alias.name not in __future__.all_feature_names:
                self.report(messages.FutureFeatureNotDefined, node, alias.name)
            if alias.name == 'annotations':
                self.annotationsFutureEnabled = True
        elif alias.name == '*':
            if not isinstance(self.scope, ModuleScope):
                self.report(messages.ImportStarNotPermitted, node, module)
                continue
            self.scope.importStarred = True
            self.report(messages.ImportStarUsed, node, module)
            importation = StarImportation(module, node)
        else:
            importation = ImportationFrom(name, node, module, alias.name)
        self.addBinding(node, importation)