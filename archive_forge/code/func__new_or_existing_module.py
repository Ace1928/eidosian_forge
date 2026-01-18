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
@contextmanager
def _new_or_existing_module(name, **kwargs):
    created_module = False
    module = sys.modules.get(name)
    try:
        if not module:
            module = ModuleType(name)
            created_module = True
            sys.modules[name] = module
        for attr, value in kwargs.items():
            setattr(module, attr, value)
        yield module
    except Exception:
        if created_module:
            if sys.modules.get(name):
                sys.modules.pop(name)
        raise