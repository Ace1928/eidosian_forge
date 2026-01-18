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
class LegacyModuleUtilLocator(ModuleUtilLocatorBase):

    def __init__(self, fq_name_parts, is_ambiguous=False, mu_paths=None, child_is_redirected=False):
        super(LegacyModuleUtilLocator, self).__init__(fq_name_parts, is_ambiguous, child_is_redirected)
        if fq_name_parts[0:2] != ('ansible', 'module_utils'):
            raise Exception('this class can only locate from ansible.module_utils, got {0}'.format(fq_name_parts))
        if fq_name_parts[2] == 'six':
            fq_name_parts = ('ansible', 'module_utils', 'six')
            self.candidate_names = [fq_name_parts]
        self._mu_paths = mu_paths
        self._collection_name = 'ansible.builtin'
        self._locate(redirect_first=False)

    def _get_module_utils_remainder_parts(self, name_parts):
        return name_parts[2:]

    def _find_module(self, name_parts):
        rel_name_parts = self._get_module_utils_remainder_parts(name_parts)
        if len(rel_name_parts) == 1:
            paths = self._mu_paths
        else:
            paths = [os.path.join(p, *rel_name_parts[:-1]) for p in self._mu_paths]
        self._info = info = importlib.machinery.PathFinder.find_spec('.'.join(name_parts), paths)
        if info is not None and os.path.splitext(info.origin)[1] in importlib.machinery.SOURCE_SUFFIXES:
            self.is_package = info.origin.endswith('/__init__.py')
            path = info.origin
        else:
            return False
        self.source_code = _slurp(path)
        return True