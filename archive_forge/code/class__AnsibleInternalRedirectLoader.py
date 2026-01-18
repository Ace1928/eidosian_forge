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
class _AnsibleInternalRedirectLoader:

    def __init__(self, fullname, path_list):
        self._redirect = None
        split_name = fullname.split('.')
        toplevel_pkg = split_name[0]
        module_to_load = split_name[-1]
        if toplevel_pkg != 'ansible':
            raise ImportError('not interested')
        builtin_meta = _get_collection_metadata('ansible.builtin')
        routing_entry = _nested_dict_get(builtin_meta, ['import_redirection', fullname])
        if routing_entry:
            self._redirect = routing_entry.get('redirect')
        if not self._redirect:
            raise ImportError('not redirected, go ask path_hook')

    def get_resource_reader(self, fullname):
        return _AnsibleTraversableResources(fullname, self)

    def exec_module(self, module):
        if not self._redirect:
            raise ValueError('no redirect found for {0}'.format(module.__spec__.name))
        sys.modules[module.__spec__.name] = import_module(self._redirect)

    def create_module(self, spec):
        return None

    def load_module(self, fullname):
        if not self._redirect:
            raise ValueError('no redirect found for {0}'.format(fullname))
        mod = import_module(self._redirect)
        sys.modules[fullname] = mod
        return mod