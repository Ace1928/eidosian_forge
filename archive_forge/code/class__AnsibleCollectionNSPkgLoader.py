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
class _AnsibleCollectionNSPkgLoader(_AnsibleCollectionPkgLoaderBase):

    def _validate_args(self):
        super(_AnsibleCollectionNSPkgLoader, self)._validate_args()
        if len(self._split_name) != 2:
            raise ImportError('this loader can only load collections namespace packages, not {0}'.format(self._fullname))

    def _validate_final(self):
        if not self._subpackage_search_paths and self._package_to_load != 'ansible':
            raise ImportError('no {0} found in {1}'.format(self._package_to_load, self._candidate_paths))