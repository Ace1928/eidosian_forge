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
@staticmethod
def _module_file_from_path(leaf_name, path):
    has_code = True
    package_path = os.path.join(to_native(path), to_native(leaf_name))
    module_path = None
    if os.path.isdir(to_bytes(package_path)):
        module_path = os.path.join(package_path, '__init__.py')
        if not os.path.isfile(to_bytes(module_path)):
            module_path = os.path.join(package_path, '__synthetic__')
            has_code = False
    else:
        module_path = package_path + '.py'
        package_path = None
        if not os.path.isfile(to_bytes(module_path)):
            raise ImportError('{0} not found at {1}'.format(leaf_name, path))
    return (module_path, has_code, package_path)