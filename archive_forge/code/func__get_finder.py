from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
def _get_finder(self, fullname):
    split_name = fullname.split('.')
    toplevel_pkg = split_name[0]
    if toplevel_pkg == 'ansible_collections':
        return self._collection_finder
    else:
        if PY3:
            if not self._file_finder:
                try:
                    self._file_finder = _AnsiblePathHookFinder._filefinder_path_hook(self._pathctx)
                except ImportError:
                    return None
            return self._file_finder
        return pkgutil.ImpImporter(self._pathctx)