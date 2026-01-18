from __future__ import (absolute_import, division, print_function)
import ast
import base64
import datetime
import json
import os
import shlex
import time
import zipfile
import re
import pkgutil
from ast import AST, Import, ImportFrom
from io import BytesIO
from ansible.release import __version__, __author__
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.executor.interpreter_discovery import InterpreterDiscoveryRequiredError
from ansible.executor.powershell import module_manifest as ps_manifest
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.plugins.loader import module_utils_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, _nested_dict_get
from ansible.executor import action_write_locks
from ansible.utils.display import Display
from collections import namedtuple
import importlib.util
import importlib.machinery
import sys
import {1} as mod
class CollectionModuleUtilLocator(ModuleUtilLocatorBase):

    def __init__(self, fq_name_parts, is_ambiguous=False, child_is_redirected=False, is_optional=False):
        super(CollectionModuleUtilLocator, self).__init__(fq_name_parts, is_ambiguous, child_is_redirected, is_optional)
        if fq_name_parts[0] != 'ansible_collections':
            raise Exception('CollectionModuleUtilLocator can only locate from ansible_collections, got {0}'.format(fq_name_parts))
        elif len(fq_name_parts) >= 6 and fq_name_parts[3:5] != ('plugins', 'module_utils'):
            raise Exception('CollectionModuleUtilLocator can only locate below ansible_collections.(ns).(coll).plugins.module_utils, got {0}'.format(fq_name_parts))
        self._collection_name = '.'.join(fq_name_parts[1:3])
        self._locate()

    def _find_module(self, name_parts):
        if len(name_parts) < 6:
            self.source_code = ''
            self.is_package = True
            return True
        collection_pkg_name = '.'.join(name_parts[0:3])
        resource_base_path = os.path.join(*name_parts[3:])
        src = None
        try:
            src = pkgutil.get_data(collection_pkg_name, to_native(os.path.join(resource_base_path, '__init__.py')))
        except ImportError:
            pass
        if src is not None:
            self.is_package = True
        else:
            try:
                src = pkgutil.get_data(collection_pkg_name, to_native(resource_base_path + '.py'))
            except ImportError:
                pass
        if src is None:
            return False
        self.source_code = src
        return True

    def _get_module_utils_remainder_parts(self, name_parts):
        return name_parts[5:]