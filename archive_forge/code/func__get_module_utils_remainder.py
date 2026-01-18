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
def _get_module_utils_remainder(self, name_parts):
    return '.'.join(self._get_module_utils_remainder_parts(name_parts))