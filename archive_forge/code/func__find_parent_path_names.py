import _imp
import _io
import sys
import _warnings
import marshal
def _find_parent_path_names(self):
    """Returns a tuple of (parent-module-name, parent-path-attr-name)"""
    parent, dot, me = self._name.rpartition('.')
    if dot == '':
        return ('sys', 'path')
    return (parent, '__path__')