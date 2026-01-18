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
def _ansible_collection_path_hook(self, path):
    path = to_native(path)
    interesting_paths = self._n_cached_collection_qualified_paths
    if not interesting_paths:
        interesting_paths = []
        for p in self._n_collection_paths:
            if os.path.basename(p) != 'ansible_collections':
                p = os.path.join(p, 'ansible_collections')
            if p not in interesting_paths:
                interesting_paths.append(p)
        interesting_paths.insert(0, self._ansible_pkg_path)
        self._n_cached_collection_qualified_paths = interesting_paths
    if any((path.startswith(p) for p in interesting_paths)):
        return _AnsiblePathHookFinder(self, path)
    raise ImportError('not interested')