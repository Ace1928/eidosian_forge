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
class Jinja2Loader(PluginLoader):
    """
    PluginLoader optimized for Jinja2 plugins

    The filter and test plugins are Jinja2 plugins encapsulated inside of our plugin format.
    We need to do a few things differently in the base class because of file == plugin
    assumptions and dedupe logic.
    """

    def __init__(self, class_name, package, config, subdir, plugin_wrapper_type, aliases=None, required_base_class=None):
        super(Jinja2Loader, self).__init__(class_name, package, config, subdir, aliases=aliases, required_base_class=required_base_class)
        self._plugin_wrapper_type = plugin_wrapper_type
        self._cached_non_collection_wrappers = {}

    def _clear_caches(self):
        super(Jinja2Loader, self)._clear_caches()
        self._cached_non_collection_wrappers = {}

    def find_plugin(self, name, mod_type='', ignore_deprecated=False, check_aliases=False, collection_list=None):
        raise NotImplementedError('find_plugin is not supported on Jinja2Loader')

    @property
    def method_map_name(self):
        return get_plugin_class(self.class_name) + 's'

    def get_contained_plugins(self, collection, plugin_path, name):
        plugins = []
        full_name = '.'.join(['ansible_collections', collection, 'plugins', self.type, name])
        try:
            if plugin_path not in self._module_cache:
                self._module_cache[plugin_path] = self._load_module_source(full_name, plugin_path)
            module = self._module_cache[plugin_path]
            obj = getattr(module, self.class_name)
        except Exception as e:
            raise KeyError('Failed to load %s for %s: %s' % (plugin_path, collection, to_native(e)))
        plugin_impl = obj()
        if plugin_impl is None:
            raise KeyError('Could not find %s.%s' % (collection, name))
        try:
            method_map = getattr(plugin_impl, self.method_map_name)
            plugin_map = method_map().items()
        except Exception as e:
            display.warning("Ignoring %s plugins in '%s' as it seems to be invalid: %r" % (self.type, to_text(plugin_path), e))
            return plugins
        for func_name, func in plugin_map:
            fq_name = '.'.join((collection, func_name))
            full = '.'.join((full_name, func_name))
            plugin = self._plugin_wrapper_type(func)
            if plugin in plugins:
                continue
            self._update_object(plugin, full, plugin_path, resolved=fq_name)
            plugins.append(plugin)
        return plugins

    def get_with_context(self, name, *args, **kwargs):
        kwargs.pop('class_only', False)
        kwargs.pop('collection_list', None)
        context = PluginLoadContext()
        name = name.removeprefix('ansible.legacy.')
        self._ensure_non_collection_wrappers(*args, **kwargs)
        if (known_plugin := self._cached_non_collection_wrappers.get(name)):
            context.resolved = True
            context.plugin_resolved_name = name
            context.plugin_resolved_path = known_plugin._original_path
            context.plugin_resolved_collection = 'ansible.builtin' if known_plugin.ansible_name.startswith('ansible.builtin.') else ''
            context._resolved_fqcn = known_plugin.ansible_name
            return get_with_context_result(known_plugin, context)
        plugin = None
        key, leaf_key = get_fqcr_and_name(name)
        seen = set()
        while True:
            if key in seen:
                raise AnsibleError('recursive collection redirect found for %r' % name, 0)
            seen.add(key)
            acr = AnsibleCollectionRef.try_parse_fqcr(key, self.type)
            if not acr:
                raise KeyError('invalid plugin name: {0}'.format(key))
            try:
                ts = _get_collection_metadata(acr.collection)
            except ValueError as e:
                raise KeyError('Invalid plugin FQCN ({0}): {1}'.format(key, to_native(e)))
            routing_entry = ts.get('plugin_routing', {}).get(self.type, {}).get(leaf_key, {})
            deprecation_entry = routing_entry.get('deprecation')
            if deprecation_entry:
                warning_text = deprecation_entry.get('warning_text')
                removal_date = deprecation_entry.get('removal_date')
                removal_version = deprecation_entry.get('removal_version')
                if not warning_text:
                    warning_text = '{0} "{1}" is deprecated'.format(self.type, key)
                display.deprecated(warning_text, version=removal_version, date=removal_date, collection_name=acr.collection)
            tombstone_entry = routing_entry.get('tombstone')
            if tombstone_entry:
                warning_text = tombstone_entry.get('warning_text')
                removal_date = tombstone_entry.get('removal_date')
                removal_version = tombstone_entry.get('removal_version')
                if not warning_text:
                    warning_text = '{0} "{1}" has been removed'.format(self.type, key)
                exc_msg = display.get_deprecation_message(warning_text, version=removal_version, date=removal_date, collection_name=acr.collection, removed=True)
                raise AnsiblePluginRemovedError(exc_msg)
            redirect = routing_entry.get('redirect', None)
            if redirect:
                if not AnsibleCollectionRef.is_valid_fqcr(redirect):
                    raise AnsibleError(f'Collection {acr.collection} contains invalid redirect for {acr.collection}.{acr.resource}: {redirect}. Redirects must use fully qualified collection names.')
                next_key, leaf_key = get_fqcr_and_name(redirect, collection=acr.collection)
                display.vvv('redirecting (type: {0}) {1}.{2} to {3}'.format(self.type, acr.collection, acr.resource, next_key))
                key = next_key
            else:
                break
        try:
            pkg = import_module(acr.n_python_package_name)
        except ImportError as e:
            raise KeyError(to_native(e))
        parent_prefix = acr.collection
        if acr.subdirs:
            parent_prefix = '{0}.{1}'.format(parent_prefix, acr.subdirs)
        try:
            for dummy, module_name, ispkg in pkgutil.iter_modules(pkg.__path__, prefix=parent_prefix + '.'):
                if ispkg:
                    continue
                try:
                    plugin_impl = super(Jinja2Loader, self).get_with_context(module_name, *args, **kwargs)
                    method_map = getattr(plugin_impl.object, self.method_map_name)
                    plugin_map = method_map().items()
                except Exception as e:
                    display.warning(f"Skipping {self.type} plugins in {module_name}'; an error occurred while loading: {e}")
                    continue
                for func_name, func in plugin_map:
                    fq_name = '.'.join((parent_prefix, func_name))
                    src_name = f'ansible_collections.{acr.collection}.plugins.{self.type}.{acr.subdirs}.{func_name}'
                    if key in (func_name, fq_name):
                        plugin = self._plugin_wrapper_type(func)
                        if plugin:
                            context = plugin_impl.plugin_load_context
                            self._update_object(plugin, src_name, plugin_impl.object._original_path, resolved=fq_name)
                            break
        except AnsiblePluginRemovedError as apre:
            raise AnsibleError(to_native(apre), 0, orig_exc=apre)
        except (AnsibleError, KeyError):
            raise
        except Exception as ex:
            display.warning('An unexpected error occurred during Jinja2 plugin loading: {0}'.format(to_native(ex)))
            display.vvv('Unexpected error during Jinja2 plugin loading: {0}'.format(format_exc()))
            raise AnsibleError(to_native(ex), 0, orig_exc=ex)
        return get_with_context_result(plugin, context)

    def all(self, *args, **kwargs):
        kwargs.pop('_dedupe', None)
        path_only = kwargs.pop('path_only', False)
        class_only = kwargs.pop('class_only', False)
        if path_only and class_only:
            raise AnsibleError('Do not set both path_only and class_only when calling PluginLoader.all()')
        self._ensure_non_collection_wrappers(*args, **kwargs)
        if path_only:
            yield from (w._original_path for w in self._cached_non_collection_wrappers.values())
        else:
            yield from (w for w in self._cached_non_collection_wrappers.values())

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