from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
class exclude_local_folder_imports(object):
    """
    A context-manager that prevents standard library modules like configparser
    from being imported from the local python-future source folder on Py3.

    (This was need prior to v0.16.0 because the presence of a configparser
    folder would otherwise have prevented setuptools from running on Py3. Maybe
    it's not needed any more?)
    """

    def __init__(self, *args):
        assert len(args) > 0
        self.module_names = args
        if any(['.' in m for m in self.module_names]):
            raise NotImplementedError('Dotted module names are not supported')

    def __enter__(self):
        self.old_sys_path = copy.copy(sys.path)
        self.old_sys_modules = copy.copy(sys.modules)
        if sys.version_info[0] < 3:
            return
        FUTURE_SOURCE_SUBFOLDERS = ['future', 'past', 'libfuturize', 'libpasteurize', 'builtins']
        for folder in self.old_sys_path:
            if all([os.path.exists(os.path.join(folder, subfolder)) for subfolder in FUTURE_SOURCE_SUBFOLDERS]):
                sys.path.remove(folder)
        for m in self.module_names:
            try:
                module = __import__(m, level=0)
            except ImportError:
                pass

    def __exit__(self, *args):
        sys.path = self.old_sys_path
        for m in set(self.old_sys_modules.keys()) - set(sys.modules.keys()):
            sys.modules[m] = self.old_sys_modules[m]