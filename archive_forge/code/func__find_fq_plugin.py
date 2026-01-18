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
def _find_fq_plugin(self, fq_name, extension, plugin_load_context, ignore_deprecated=False):
    """Search builtin paths to find a plugin. No external paths are searched,
        meaning plugins inside roles inside collections will be ignored.
        """
    plugin_load_context.resolved = False
    plugin_type = AnsibleCollectionRef.legacy_plugin_dir_to_plugin_type(self.subdir)
    acr = AnsibleCollectionRef.from_fqcr(fq_name, plugin_type)
    routing_metadata = self._query_collection_routing_meta(acr, plugin_type, extension=extension)
    action_plugin = None
    if routing_metadata:
        deprecation = routing_metadata.get('deprecation', None)
        if not ignore_deprecated:
            plugin_load_context.record_deprecation(fq_name, deprecation, acr.collection)
        tombstone = routing_metadata.get('tombstone', None)
        if tombstone:
            removal_date = tombstone.get('removal_date')
            removal_version = tombstone.get('removal_version')
            warning_text = tombstone.get('warning_text') or ''
            warning_text = '{0} has been removed.{1}{2}'.format(fq_name, ' ' if warning_text else '', warning_text)
            removed_msg = display.get_deprecation_message(msg=warning_text, version=removal_version, date=removal_date, removed=True, collection_name=acr.collection)
            plugin_load_context.removal_date = removal_date
            plugin_load_context.removal_version = removal_version
            plugin_load_context.resolved = True
            plugin_load_context.exit_reason = removed_msg
            raise AnsiblePluginRemovedError(removed_msg, plugin_load_context=plugin_load_context)
        redirect = routing_metadata.get('redirect', None)
        if redirect:
            if not AnsibleCollectionRef.is_valid_fqcr(redirect):
                raise AnsibleError(f'Collection {acr.collection} contains invalid redirect for {fq_name}: {redirect}. Redirects must use fully qualified collection names.')
            display.vv('redirecting (type: {0}) {1} to {2}'.format(plugin_type, fq_name, redirect))
            if fq_name not in plugin_load_context.redirect_list:
                plugin_load_context.redirect_list.append(fq_name)
            return plugin_load_context.redirect(redirect)
        if self.type == 'modules':
            action_plugin = routing_metadata.get('action_plugin')
    n_resource = to_native(acr.resource, errors='strict')
    full_name = '{0}.{1}'.format(acr.n_python_package_name, n_resource)
    if extension:
        n_resource += extension
    pkg = sys.modules.get(acr.n_python_package_name)
    if not pkg:
        try:
            pkg = import_module(acr.n_python_package_name)
        except ImportError:
            return plugin_load_context.nope('Python package {0} not found'.format(acr.n_python_package_name))
    pkg_path = os.path.dirname(pkg.__file__)
    n_resource_path = os.path.join(pkg_path, n_resource)
    if os.path.exists(n_resource_path):
        return plugin_load_context.resolve(full_name, to_text(n_resource_path), acr.collection, 'found exact match for {0} in {1}'.format(full_name, acr.collection), action_plugin)
    if extension:
        return plugin_load_context.nope('no match for {0} in {1}'.format(to_text(n_resource), acr.collection))
    found_files = [f for f in glob.iglob(os.path.join(pkg_path, n_resource) + '.*') if os.path.isfile(f) and (not f.endswith(C.MODULE_IGNORE_EXTS))]
    if not found_files:
        return plugin_load_context.nope('failed fuzzy extension match for {0} in {1}'.format(full_name, acr.collection))
    found_files = sorted(found_files)
    if len(found_files) > 1:
        display.debug('Found several possible candidates for the plugin but using first: %s' % ','.join(found_files))
    return plugin_load_context.resolve(full_name, to_text(found_files[0]), acr.collection, 'found fuzzy extension match for {0} in {1}'.format(full_name, acr.collection), action_plugin)