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
def _GetModuleFromPathViaPkgutil(module_path, name_to_give):
    """Loads module by using pkgutil.get_importer mechanism."""
    importer = pkgutil.get_importer(os.path.dirname(module_path))
    if not importer:
        raise ImportError('{0} not found'.format(module_path))
    find_spec_exists = hasattr(importer, 'find_spec')
    find_method = importer.find_spec if find_spec_exists else importer.find_module
    module_name = os.path.basename(module_path)
    if find_method(module_name):
        return _LoadModule(importer, module_path, module_name, name_to_give)
    raise ImportError('{0} not found'.format(module_path))