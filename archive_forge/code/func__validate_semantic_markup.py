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
def _validate_semantic_markup(self, object) -> None:
    if is_iterable(object):
        for entry in object:
            self._validate_semantic_markup(entry)
        return
    if not isinstance(object, string_types):
        return
    if self.collection:
        fqcn = f'{self.collection_name}.{self.name}'
    else:
        fqcn = f'ansible.builtin.{self.name}'
    current_plugin = dom.PluginIdentifier(fqcn=fqcn, type=self.plugin_type)
    for par in parse(object, Context(current_plugin=current_plugin), errors='message', add_source=True):
        for part in par:
            if part.type == dom.PartType.OPTION_NAME:
                self._check_sem_option(part, current_plugin)
            if part.type == dom.PartType.RETURN_VALUE:
                self._check_sem_return_value(part, current_plugin)