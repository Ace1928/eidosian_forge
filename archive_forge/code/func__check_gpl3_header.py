from __future__ import annotations
import abc
import argparse
import ast
import datetime
import json
import os
import re
import sys
import traceback
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from fnmatch import fnmatch
from antsibull_docs_parser import dom
from antsibull_docs_parser.parser import parse, Context
import yaml
from voluptuous.humanize import humanize_error
from ansible import __version__ as ansible_version
from ansible.executor.module_common import REPLACER_WINDOWS, NEW_STYLE_PYTHON_MODULE_RE
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.parameters import DEFAULT_TYPE_VALIDATORS
from ansible.module_utils.compat.version import StrictVersion, LooseVersion
from ansible.module_utils.basic import to_bytes
from ansible.module_utils.six import PY3, with_metaclass, string_types
from ansible.plugins.loader import fragment_loader
from ansible.plugins.list import IGNORE as REJECTLIST
from ansible.utils.plugin_docs import add_collection_to_versions_and_dates, add_fragments, get_docstring
from ansible.utils.version import SemanticVersion
from .module_args import AnsibleModuleImportError, AnsibleModuleNotInitialized, get_argument_spec
from .schema import (
from .utils import CaptureStd, NoArgsAnsibleModule, compare_unordered_lists, parse_yaml, parse_isodate
def _check_gpl3_header(self):
    header = '\n'.join(self.text.split('\n')[:20])
    if 'GNU General Public License' not in header or ('version 3' not in header and 'v3.0' not in header):
        self.reporter.error(path=self.object_path, code='missing-gplv3-license', msg='GPLv3 license header not found in the first 20 lines of the module')
    elif self._is_new_module():
        if len([line for line in header if 'GNU General Public License' in line]) > 1:
            self.reporter.error(path=self.object_path, code='use-short-gplv3-license', msg='Found old style GPLv3 license header: https://docs.ansible.com/ansible-core/devel/dev_guide/developing_modules_documenting.html#copyright')