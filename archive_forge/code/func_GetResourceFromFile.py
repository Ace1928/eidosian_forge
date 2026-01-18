from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import glob
import importlib.util
import os
import pkgutil
import sys
import types
from googlecloudsdk.core.util import files
def GetResourceFromFile(path):
    """Gets the given resource as a byte string.

  This is similar to GetResource(), but uses file paths instead of module names.

  Args:
    path: str, filesystem like path to a file/resource.

  Returns:
    The contents of the resource as a byte string.

  Raises:
    IOError: if resource is not found under given path.
  """
    if os.path.isfile(path):
        return files.ReadBinaryFileContents(path)
    importer = pkgutil.get_importer(os.path.dirname(path))
    if hasattr(importer, 'get_data'):
        return importer.get_data(path)
    raise IOError('File not found {0}'.format(path))