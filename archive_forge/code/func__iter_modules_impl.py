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
def _iter_modules_impl(paths, prefix=''):
    if not prefix:
        prefix = ''
    else:
        prefix = to_native(prefix)
    for b_path in map(to_bytes, paths):
        if not os.path.isdir(b_path):
            continue
        for b_basename in sorted(os.listdir(b_path)):
            b_candidate_module_path = os.path.join(b_path, b_basename)
            if os.path.isdir(b_candidate_module_path):
                if b'.' in b_basename or b_basename == b'__pycache__':
                    continue
                yield (prefix + to_native(b_basename), True)
            elif b_basename.endswith(b'.py') and b_basename != b'__init__.py':
                yield (prefix + to_native(os.path.splitext(b_basename)[0]), False)