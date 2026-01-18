from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
class _AnsibleCollectionFinder:

    def __init__(self, paths=None, scan_sys_paths=True):
        self._ansible_pkg_path = to_native(os.path.dirname(to_bytes(sys.modules['ansible'].__file__)))
        if isinstance(paths, string_types):
            paths = [paths]
        elif paths is None:
            paths = []
        paths = [os.path.expanduser(to_native(p, errors='surrogate_or_strict')) for p in paths]
        if scan_sys_paths:
            paths.extend(sys.path)
        good_paths = []
        for p in paths:
            if os.path.basename(p) == 'ansible_collections':
                p = os.path.dirname(p)
            if p not in good_paths and os.path.isdir(to_bytes(os.path.join(p, 'ansible_collections'), errors='surrogate_or_strict')):
                good_paths.append(p)
        self._n_configured_paths = good_paths
        self._n_cached_collection_paths = None
        self._n_cached_collection_qualified_paths = None
        self._n_playbook_paths = []

    @classmethod
    def _remove(cls):
        for mps in sys.meta_path:
            if isinstance(mps, _AnsibleCollectionFinder):
                sys.meta_path.remove(mps)
        for ph in sys.path_hooks:
            if hasattr(ph, '__self__') and isinstance(ph.__self__, _AnsibleCollectionFinder):
                sys.path_hooks.remove(ph)
        sys.path_importer_cache.clear()
        AnsibleCollectionConfig._collection_finder = None
        if AnsibleCollectionConfig.collection_finder is not None:
            raise AssertionError('_AnsibleCollectionFinder remove did not reset AnsibleCollectionConfig.collection_finder')

    def _install(self):
        self._remove()
        sys.meta_path.insert(0, self)
        sys.path_hooks.insert(0, self._ansible_collection_path_hook)
        AnsibleCollectionConfig.collection_finder = self

    def _ansible_collection_path_hook(self, path):
        path = to_native(path)
        interesting_paths = self._n_cached_collection_qualified_paths
        if not interesting_paths:
            interesting_paths = []
            for p in self._n_collection_paths:
                if os.path.basename(p) != 'ansible_collections':
                    p = os.path.join(p, 'ansible_collections')
                if p not in interesting_paths:
                    interesting_paths.append(p)
            interesting_paths.insert(0, self._ansible_pkg_path)
            self._n_cached_collection_qualified_paths = interesting_paths
        if any((path.startswith(p) for p in interesting_paths)):
            return _AnsiblePathHookFinder(self, path)
        raise ImportError('not interested')

    @property
    def _n_collection_paths(self):
        paths = self._n_cached_collection_paths
        if not paths:
            self._n_cached_collection_paths = paths = self._n_playbook_paths + self._n_configured_paths
        return paths

    def set_playbook_paths(self, playbook_paths):
        if isinstance(playbook_paths, string_types):
            playbook_paths = [playbook_paths]
        added_paths = set()
        self._n_playbook_paths = [os.path.join(to_native(p), 'collections') for p in playbook_paths if not (p in added_paths or added_paths.add(p))]
        self._n_cached_collection_paths = None
        for pkg in ['ansible_collections', 'ansible_collections.ansible']:
            self._reload_hack(pkg)

    def _reload_hack(self, fullname):
        m = sys.modules.get(fullname)
        if not m:
            return
        reload_module(m)

    def _get_loader(self, fullname, path=None):
        split_name = fullname.split('.')
        toplevel_pkg = split_name[0]
        module_to_find = split_name[-1]
        part_count = len(split_name)
        if toplevel_pkg not in ['ansible', 'ansible_collections']:
            return None
        if part_count == 1:
            if path:
                raise ValueError('path should not be specified for top-level packages (trying to find {0})'.format(fullname))
            else:
                path = self._n_collection_paths
        if part_count > 1 and path is None:
            raise ValueError('path must be specified for subpackages (trying to find {0})'.format(fullname))
        if toplevel_pkg == 'ansible':
            initialize_loader = _AnsibleInternalRedirectLoader
        elif part_count == 1:
            initialize_loader = _AnsibleCollectionRootPkgLoader
        elif part_count == 2:
            initialize_loader = _AnsibleCollectionNSPkgLoader
        elif part_count == 3:
            initialize_loader = _AnsibleCollectionPkgLoader
        else:
            initialize_loader = _AnsibleCollectionLoader
        try:
            return initialize_loader(fullname=fullname, path_list=path)
        except ImportError:
            return None

    def find_module(self, fullname, path=None):
        return self._get_loader(fullname, path)

    def find_spec(self, fullname, path, target=None):
        loader = self._get_loader(fullname, path)
        if loader is None:
            return None
        spec = spec_from_loader(fullname, loader)
        if spec is not None and hasattr(loader, '_subpackage_search_paths'):
            spec.submodule_search_locations = loader._subpackage_search_paths
        return spec