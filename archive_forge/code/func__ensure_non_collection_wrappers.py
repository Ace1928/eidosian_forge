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
def _ensure_non_collection_wrappers(self, *args, **kwargs):
    if self._cached_non_collection_wrappers:
        return
    for p_map in super(Jinja2Loader, self).all(*args, **kwargs):
        is_builtin = p_map.ansible_name.startswith('ansible.builtin.')
        plugins_list = getattr(p_map, self.method_map_name)
        try:
            plugins = plugins_list()
        except Exception as e:
            display.vvvv("Skipping %s plugins in '%s' as it seems to be invalid: %r" % (self.type, to_text(p_map._original_path), e))
            continue
        for plugin_name in plugins.keys():
            if '.' in plugin_name:
                display.debug(f'{plugin_name} skipped in {p_map._original_path}; Jinja plugin short names may not contain "."')
                continue
            if plugin_name in _PLUGIN_FILTERS[self.package]:
                display.debug('%s skipped due to a defined plugin filter' % plugin_name)
                continue
            wrapper = self._plugin_wrapper_type(plugins[plugin_name])
            fqcn = plugin_name
            collection = '.'.join(p_map.ansible_name.split('.')[:2]) if p_map.ansible_name.count('.') >= 2 else ''
            if not plugin_name.startswith(collection):
                fqcn = f'{collection}.{plugin_name}'
            self._update_object(wrapper, plugin_name, p_map._original_path, resolved=fqcn)
            target_names = {plugin_name, fqcn}
            if is_builtin:
                target_names.add(f'ansible.builtin.{plugin_name}')
            for target_name in target_names:
                if (existing_plugin := self._cached_non_collection_wrappers.get(target_name)):
                    display.debug(f'Jinja plugin {target_name} from {p_map._original_path} skipped; shadowed by plugin from {existing_plugin._original_path})')
                    continue
                self._cached_non_collection_wrappers[target_name] = wrapper