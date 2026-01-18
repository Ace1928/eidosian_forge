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
def _get_subpackage_search_paths(self, candidate_paths):
    collection_name = '.'.join(self._split_name[1:3])
    collection_meta = _get_collection_metadata(collection_name)
    redirect = None
    explicit_redirect = False
    routing_entry = _nested_dict_get(collection_meta, ['import_redirection', self._fullname])
    if routing_entry:
        redirect = routing_entry.get('redirect')
    if redirect:
        explicit_redirect = True
    else:
        redirect = _get_ancestor_redirect(self._redirected_package_map, self._fullname)
    if redirect:
        self._redirect_module = import_module(redirect)
        if explicit_redirect and hasattr(self._redirect_module, '__path__') and self._redirect_module.__path__:
            self._redirected_package_map[self._fullname] = redirect
        return None
    if not candidate_paths:
        raise ImportError('package has no paths')
    found_path, has_code, package_path = self._module_file_from_path(self._package_to_load, candidate_paths[0])
    if has_code:
        self._source_code_path = found_path
    if package_path:
        return [package_path]
    return None