from __future__ import (absolute_import, division, print_function)
import ast
import datetime
import os
import pwd
import re
import time
from collections.abc import Iterator, Sequence, Mapping, MappingView, MutableMapping
from contextlib import contextmanager
from numbers import Number
from traceback import format_exc
from jinja2.exceptions import TemplateSyntaxError, UndefinedError, SecurityError
from jinja2.loaders import FileSystemLoader
from jinja2.nativetypes import NativeEnvironment
from jinja2.runtime import Context, StrictUndefined
from ansible import constants as C
from ansible.errors import (
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.collections import is_sequence
from ansible.plugins.loader import filter_loader, lookup_loader, test_loader
from ansible.template.native_helpers import ansible_native_concat, ansible_eval_concat, ansible_concat
from ansible.template.template import AnsibleJ2Template
from ansible.template.vars import AnsibleJ2Vars
from ansible.utils.display import Display
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.native_jinja import NativeJinjaText
from ansible.utils.unsafe_proxy import to_unsafe_text, wrap_var, AnsibleUnsafeText, AnsibleUnsafeBytes, NativeJinjaUnsafeText
def _convert_bare_variable(self, variable):
    """
        Wraps a bare string, which may have an attribute portion (ie. foo.bar)
        in jinja2 variable braces so that it is evaluated properly.
        """
    if isinstance(variable, string_types):
        contains_filters = '|' in variable
        first_part = variable.split('|')[0].split('.')[0].split('[')[0]
        if (contains_filters or first_part in self._available_variables) and self.environment.variable_start_string not in variable:
            return '%s%s%s' % (self.environment.variable_start_string, variable, self.environment.variable_end_string)
    return variable