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
def _get_filefinder_path_hook(self=None):
    _file_finder_hook = None
    if PY3:
        _file_finder_hook = [ph for ph in sys.path_hooks if 'FileFinder' in repr(ph)]
        if len(_file_finder_hook) != 1:
            raise Exception('need exactly one FileFinder import hook (found {0})'.format(len(_file_finder_hook)))
        _file_finder_hook = _file_finder_hook[0]
    return _file_finder_hook