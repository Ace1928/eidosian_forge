from __future__ import (absolute_import, division, print_function)
import glob
import os
import os.path
import pkgutil
import sys
import warnings
from collections import defaultdict, namedtuple
from traceback import format_exc
import ansible.module_utils.compat.typing as t
from .filter import AnsibleJinja2Filter
from .test import AnsibleJinja2Test
from ansible import __version__ as ansible_version
from ansible import constants as C
from ansible.errors import AnsibleError, AnsiblePluginCircularRedirect, AnsiblePluginRemovedError, AnsibleCollectionUnsupportedVersionError
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.module_utils.compat.importlib import import_module
from ansible.module_utils.six import string_types
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.plugins import get_plugin_class, MODULE_CACHE, PATH_CACHE, PLUGIN_PATH_CACHE
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionFinder, _get_collection_metadata
from ansible.utils.display import Display
from ansible.utils.plugin_docs import add_fragments
from ansible.utils.unsafe_proxy import _is_unsafe
import importlib.util
class PluginLoadContext(object):

    def __init__(self):
        self.original_name = None
        self.redirect_list = []
        self.error_list = []
        self.import_error_list = []
        self.load_attempts = []
        self.pending_redirect = None
        self.exit_reason = None
        self.plugin_resolved_path = None
        self.plugin_resolved_name = None
        self.plugin_resolved_collection = None
        self.deprecated = False
        self.removal_date = None
        self.removal_version = None
        self.deprecation_warnings = []
        self.resolved = False
        self._resolved_fqcn = None
        self.action_plugin = None

    @property
    def resolved_fqcn(self):
        if not self.resolved:
            return
        if not self._resolved_fqcn:
            final_plugin = self.redirect_list[-1]
            if AnsibleCollectionRef.is_valid_fqcr(final_plugin) and final_plugin.startswith('ansible.legacy.'):
                final_plugin = final_plugin.split('ansible.legacy.')[-1]
            if self.plugin_resolved_collection and (not AnsibleCollectionRef.is_valid_fqcr(final_plugin)):
                final_plugin = self.plugin_resolved_collection + '.' + final_plugin
            self._resolved_fqcn = final_plugin
        return self._resolved_fqcn

    def record_deprecation(self, name, deprecation, collection_name):
        if not deprecation:
            return self
        warning_text = deprecation.get('warning_text', None) or ''
        removal_date = deprecation.get('removal_date', None)
        removal_version = deprecation.get('removal_version', None)
        if removal_date is not None:
            removal_version = None
        warning_text = '{0} has been deprecated.{1}{2}'.format(name, ' ' if warning_text else '', warning_text)
        display.deprecated(warning_text, date=removal_date, version=removal_version, collection_name=collection_name)
        self.deprecated = True
        if removal_date:
            self.removal_date = removal_date
        if removal_version:
            self.removal_version = removal_version
        self.deprecation_warnings.append(warning_text)
        return self

    def resolve(self, resolved_name, resolved_path, resolved_collection, exit_reason, action_plugin):
        self.pending_redirect = None
        self.plugin_resolved_name = resolved_name
        self.plugin_resolved_path = resolved_path
        self.plugin_resolved_collection = resolved_collection
        self.exit_reason = exit_reason
        self.resolved = True
        self.action_plugin = action_plugin
        return self

    def redirect(self, redirect_name):
        self.pending_redirect = redirect_name
        self.exit_reason = 'pending redirect resolution from {0} to {1}'.format(self.original_name, redirect_name)
        self.resolved = False
        return self

    def nope(self, exit_reason):
        self.pending_redirect = None
        self.exit_reason = exit_reason
        self.resolved = False
        return self