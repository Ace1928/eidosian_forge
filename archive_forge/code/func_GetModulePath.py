from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import importlib
import importlib.util
import os
import sys
from googlecloudsdk.core import exceptions
import six
def GetModulePath(obj):
    """Returns the module path string for obj, None if it's builtin.

  The module path is relative and importable by ImportModule() from any
  installation of the current release.

  Args:
    obj: The object to get the module path from.

  Returns:
    The module path name for obj if not builtin else None.
  """
    try:
        module = obj.__module__
    except AttributeError:
        obj = obj.__class__
        module = obj.__module__
    if six.PY3 and module == 'builtins':
        return None
    if module.startswith('__'):
        module = _GetPrivateModulePath(module)
        if not module:
            return None
    try:
        return module + ':' + obj.__name__
    except AttributeError:
        try:
            return module + ':' + obj.__class__.__name__
        except AttributeError:
            return None