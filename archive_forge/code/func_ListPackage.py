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
def ListPackage(path, extra_extensions=None):
    """Returns list of packages and modules in given path.

  Args:
    path: str, filesystem path
    extra_extensions: [str], The list of file extra extensions that should be
      considered modules for the purposes of listing (in addition to .py).

  Returns:
    tuple([packages], [modules])
  """
    iter_modules = []
    if os.path.isdir(path):
        iter_modules = _IterModules(_ListPackagesAndFiles(path), extra_extensions)
    else:
        importer = pkgutil.get_importer(path)
        if hasattr(importer, '_files'):
            iter_modules = _IterModules(importer._files, extra_extensions, importer.prefix)
    packages, modules = ([], [])
    for name, ispkg in iter_modules:
        if ispkg:
            packages.append(name)
        else:
            modules.append(name)
    return (sorted(packages), sorted(modules))