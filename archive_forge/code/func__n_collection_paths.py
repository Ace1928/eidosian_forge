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
@property
def _n_collection_paths(self):
    paths = self._n_cached_collection_paths
    if not paths:
        self._n_cached_collection_paths = paths = self._n_playbook_paths + self._n_configured_paths
    return paths