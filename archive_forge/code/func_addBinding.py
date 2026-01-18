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
def addBinding(self, node, value):
    """
        Called when a binding is altered.

        - `node` is the statement responsible for the change
        - `value` is the new value, a Binding instance
        """
    for scope in self.scopeStack[::-1]:
        if value.name in scope:
            break
    existing = scope.get(value.name)
    if existing and (not isinstance(existing, Builtin)) and (not self.differentForks(node, existing.source)):
        parent_stmt = self.getParent(value.source)
        if isinstance(existing, Importation) and isinstance(parent_stmt, FOR_TYPES):
            self.report(messages.ImportShadowedByLoopVar, node, value.name, existing.source)
        elif scope is self.scope:
            if (not existing.used and value.redefines(existing)) and (value.name != '_' or isinstance(existing, Importation)) and (not is_typing_overload(existing, self.scopeStack)):
                self.report(messages.RedefinedWhileUnused, node, value.name, existing.source)
        elif isinstance(existing, Importation) and value.redefines(existing):
            existing.redefined.append(node)
    if value.name in self.scope:
        value.used = self.scope[value.name].used
    if value.name not in self.scope or not isinstance(value, Annotation):
        cur_scope_pos = -1
        while isinstance(value, NamedExprAssignment) and isinstance(self.scopeStack[cur_scope_pos], GeneratorScope):
            cur_scope_pos -= 1
        self.scopeStack[cur_scope_pos][value.name] = value