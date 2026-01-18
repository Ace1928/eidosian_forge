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
def _resolve_plugin_step(self, name, mod_type='', ignore_deprecated=False, check_aliases=False, collection_list=None, plugin_load_context=PluginLoadContext()):
    if not plugin_load_context:
        raise ValueError('A PluginLoadContext is required')
    plugin_load_context.redirect_list.append(name)
    plugin_load_context.resolved = False
    if name in _PLUGIN_FILTERS[self.package]:
        plugin_load_context.exit_reason = '{0} matched a defined plugin filter'.format(name)
        return plugin_load_context
    if mod_type:
        suffix = mod_type
    elif self.class_name:
        suffix = '.py'
    else:
        suffix = ''
    if (AnsibleCollectionRef.is_valid_fqcr(name) or collection_list) and (not name.startswith('Ansible')):
        if '.' in name or not collection_list:
            candidates = [name]
        else:
            candidates = ['{0}.{1}'.format(c, name) for c in collection_list]
        for candidate_name in candidates:
            try:
                plugin_load_context.load_attempts.append(candidate_name)
                if candidate_name.startswith('ansible.legacy'):
                    plugin_load_context = self._find_plugin_legacy(name.removeprefix('ansible.legacy.'), plugin_load_context, ignore_deprecated, check_aliases, suffix)
                else:
                    plugin_load_context = self._find_fq_plugin(candidate_name, suffix, plugin_load_context=plugin_load_context, ignore_deprecated=ignore_deprecated)
                    if plugin_load_context.resolved and candidate_name not in plugin_load_context.redirect_list:
                        plugin_load_context.redirect_list.append(candidate_name)
                if plugin_load_context.resolved or plugin_load_context.pending_redirect:
                    return plugin_load_context
            except (AnsiblePluginRemovedError, AnsiblePluginCircularRedirect, AnsibleCollectionUnsupportedVersionError):
                raise
            except ImportError as ie:
                plugin_load_context.import_error_list.append(ie)
            except Exception as ex:
                plugin_load_context.error_list.append(to_native(ex))
        if plugin_load_context.error_list:
            display.debug(msg='plugin lookup for {0} failed; errors: {1}'.format(name, '; '.join(plugin_load_context.error_list)))
        plugin_load_context.exit_reason = 'no matches found for {0}'.format(name)
        return plugin_load_context
    return self._find_plugin_legacy(name, plugin_load_context, ignore_deprecated, check_aliases, suffix)