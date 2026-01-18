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
def _IterModules(file_list, extra_extensions, prefix=None):
    """Yields module names from given list of file paths with given prefix."""
    yielded = set()
    if extra_extensions is None:
        extra_extensions = []
    if prefix is None:
        prefix = ''
    for file_path in file_list:
        if not file_path.startswith(prefix):
            continue
        file_path_parts = file_path[len(prefix):].split(os.sep)
        if len(file_path_parts) == 2 and file_path_parts[1].startswith('__init__.py'):
            if file_path_parts[0] not in yielded:
                yielded.add(file_path_parts[0])
                yield (file_path_parts[0], True)
        if len(file_path_parts) != 1:
            continue
        filename = os.path.basename(file_path_parts[0])
        modname, ext = os.path.splitext(filename)
        if modname == '__init__' or (ext != '.py' and ext not in extra_extensions):
            continue
        to_yield = modname if ext == '.py' else filename
        if '.' not in modname and to_yield not in yielded:
            yielded.add(to_yield)
            yield (to_yield, False)