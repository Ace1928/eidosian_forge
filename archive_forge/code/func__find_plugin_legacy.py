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
def _find_plugin_legacy(self, name, plugin_load_context, ignore_deprecated=False, check_aliases=False, suffix=None):
    """Search library and various *_plugins paths in order to find the file.
        This was behavior prior to the existence of collections.
        """
    plugin_load_context.resolved = False
    if check_aliases:
        name = self.aliases.get(name, name)
    pull_cache = self._plugin_path_cache[suffix]
    try:
        path_with_context = pull_cache[name]
        plugin_load_context.plugin_resolved_path = path_with_context.path
        plugin_load_context.plugin_resolved_name = name
        plugin_load_context.plugin_resolved_collection = 'ansible.builtin' if path_with_context.internal else ''
        plugin_load_context._resolved_fqcn = 'ansible.builtin.' + name if path_with_context.internal else name
        plugin_load_context.resolved = True
        return plugin_load_context
    except KeyError:
        pass
    for path_with_context in (p for p in self._get_paths_with_context() if p.path not in self._searched_paths and os.path.isdir(to_bytes(p.path))):
        path = path_with_context.path
        b_path = to_bytes(path)
        display.debug('trying %s' % path)
        plugin_load_context.load_attempts.append(path)
        internal = path_with_context.internal
        try:
            full_paths = (os.path.join(b_path, f) for f in os.listdir(b_path))
        except OSError as e:
            display.warning('Error accessing plugin paths: %s' % to_text(e))
        for full_path in (to_native(f) for f in full_paths if os.path.isfile(f) and (not f.endswith(b'__init__.py'))):
            full_name = os.path.basename(full_path)
            if any((full_path.endswith(x) for x in C.MODULE_IGNORE_EXTS)):
                continue
            splitname = os.path.splitext(full_name)
            base_name = splitname[0]
            try:
                extension = splitname[1]
            except IndexError:
                extension = ''
            full_path = to_text(full_path, errors='surrogate_or_strict')
            if base_name not in self._plugin_path_cache['']:
                self._plugin_path_cache[''][base_name] = PluginPathContext(full_path, internal)
            if full_name not in self._plugin_path_cache['']:
                self._plugin_path_cache[''][full_name] = PluginPathContext(full_path, internal)
            if base_name not in self._plugin_path_cache[extension]:
                self._plugin_path_cache[extension][base_name] = PluginPathContext(full_path, internal)
            if full_name not in self._plugin_path_cache[extension]:
                self._plugin_path_cache[extension][full_name] = PluginPathContext(full_path, internal)
        self._searched_paths.add(path)
        try:
            path_with_context = pull_cache[name]
            plugin_load_context.plugin_resolved_path = path_with_context.path
            plugin_load_context.plugin_resolved_name = name
            plugin_load_context.plugin_resolved_collection = 'ansible.builtin' if path_with_context.internal else ''
            plugin_load_context._resolved_fqcn = 'ansible.builtin.' + name if path_with_context.internal else name
            plugin_load_context.resolved = True
            return plugin_load_context
        except KeyError:
            pass
    if not name.startswith('_'):
        alias_name = '_' + name
        if alias_name in pull_cache:
            path_with_context = pull_cache[alias_name]
            if not ignore_deprecated and (not os.path.islink(path_with_context.path)):
                display.deprecated('%s is kept for backwards compatibility but usage is discouraged. The module documentation details page may explain more about this rationale.' % name.lstrip('_'))
            plugin_load_context.plugin_resolved_path = path_with_context.path
            plugin_load_context.plugin_resolved_name = alias_name
            plugin_load_context.plugin_resolved_collection = 'ansible.builtin' if path_with_context.internal else ''
            plugin_load_context._resolved_fqcn = 'ansible.builtin.' + alias_name if path_with_context.internal else alias_name
            plugin_load_context.resolved = True
            return plugin_load_context
    candidate_fqcr = 'ansible.builtin.{0}'.format(name)
    if '.' not in name and AnsibleCollectionRef.is_valid_fqcr(candidate_fqcr):
        return self._find_fq_plugin(fq_name=candidate_fqcr, extension=suffix, plugin_load_context=plugin_load_context, ignore_deprecated=ignore_deprecated)
    return plugin_load_context.nope('{0} is not eligible for last-chance resolution'.format(name))