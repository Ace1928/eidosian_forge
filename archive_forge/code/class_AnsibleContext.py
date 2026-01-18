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
class AnsibleContext(Context):
    """
    A custom context, which intercepts resolve_or_missing() calls and sets a flag
    internally if any variable lookup returns an AnsibleUnsafe value. This
    flag is checked post-templating, and (when set) will result in the
    final templated result being wrapped in AnsibleUnsafe.
    """
    _disallowed_callables = frozenset({AnsibleUnsafeText._strip_unsafe.__qualname__, AnsibleUnsafeBytes._strip_unsafe.__qualname__, NativeJinjaUnsafeText._strip_unsafe.__qualname__})

    def __init__(self, *args, **kwargs):
        super(AnsibleContext, self).__init__(*args, **kwargs)
        self.unsafe = False

    def call(self, obj, *args, **kwargs):
        if getattr(obj, '__qualname__', None) in self._disallowed_callables or obj in self._disallowed_callables:
            raise SecurityError(f'{obj!r} is not safely callable')
        return super().call(obj, *args, **kwargs)

    def _is_unsafe(self, val):
        """
        Our helper function, which will also recursively check dict and
        list entries due to the fact that they may be repr'd and contain
        a key or value which contains jinja2 syntax and would otherwise
        lose the AnsibleUnsafe value.
        """
        if isinstance(val, dict):
            for key in val.keys():
                if self._is_unsafe(val[key]):
                    return True
        elif isinstance(val, list):
            for item in val:
                if self._is_unsafe(item):
                    return True
        elif getattr(val, '__UNSAFE__', False) is True:
            return True
        return False

    def _update_unsafe(self, val):
        if val is not None and (not self.unsafe) and self._is_unsafe(val):
            self.unsafe = True

    def resolve_or_missing(self, key):
        val = super(AnsibleContext, self).resolve_or_missing(key)
        self._update_unsafe(val)
        return val

    def get_all(self):
        """Return the complete context as a dict including the exported
        variables. For optimizations reasons this might not return an
        actual copy so be careful with using it.

        This is to prevent from running ``AnsibleJ2Vars`` through dict():

            ``dict(self.parent, **self.vars)``

        In Ansible this means that ALL variables would be templated in the
        process of re-creating the parent because ``AnsibleJ2Vars`` templates
        each variable in its ``__getitem__`` method. Instead we re-create the
        parent via ``AnsibleJ2Vars.add_locals`` that creates a new
        ``AnsibleJ2Vars`` copy without templating each variable.

        This will prevent unnecessarily templating unused variables in cases
        like setting a local variable and passing it to {% include %}
        in a template.

        Also see ``AnsibleJ2Template``and
        https://github.com/pallets/jinja/commit/d67f0fd4cc2a4af08f51f4466150d49da7798729
        """
        if not self.vars:
            return self.parent
        if not self.parent:
            return self.vars
        if isinstance(self.parent, AnsibleJ2Vars):
            return self.parent.add_locals(self.vars)
        else:
            return dict(self.parent, **self.vars)