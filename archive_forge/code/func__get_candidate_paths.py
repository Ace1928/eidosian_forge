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
def _get_candidate_paths(self, path_list):
    if len(path_list) != 1 and self._split_name[1:3] != ['ansible', 'builtin']:
        raise ValueError('this loader requires exactly one path to search')
    return path_list