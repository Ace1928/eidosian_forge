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
class _AnsibleTraversableResources(TraversableResources):
    """Implements ``importlib.resources.abc.TraversableResources`` for the
    collection Python loaders.

    The result of ``files`` will depend on whether a particular collection, or
    a sub package of a collection was referenced, as opposed to
    ``ansible_collections`` or a particular namespace. For a collection and
    its subpackages, a ``pathlib.Path`` instance will be returned, whereas
    for the higher level namespace packages, ``_AnsibleNSTraversable``
    will be returned.
    """

    def __init__(self, package, loader):
        self._package = package
        self._loader = loader

    def _get_name(self, package):
        try:
            return package.name
        except AttributeError:
            return package.__name__

    def _get_package(self, package):
        try:
            return package.__parent__
        except AttributeError:
            return package.__package__

    def _get_path(self, package):
        try:
            return package.origin
        except AttributeError:
            return package.__file__

    def _is_ansible_ns_package(self, package):
        origin = getattr(package, 'origin', None)
        if not origin:
            return False
        if origin == SYNTHETIC_PACKAGE_NAME:
            return True
        module_filename = os.path.basename(origin)
        return module_filename in {'__synthetic__', '__init__.py'}

    def _ensure_package(self, package):
        if self._is_ansible_ns_package(package):
            return
        if self._get_package(package) != package.__name__:
            raise TypeError('%r is not a package' % package.__name__)

    def files(self):
        package = self._package
        parts = package.split('.')
        is_ns = parts[0] == 'ansible_collections' and len(parts) < 3
        if isinstance(package, string_types):
            if is_ns:
                package = find_spec(package)
            else:
                package = spec_from_loader(package, self._loader)
        elif not isinstance(package, ModuleType):
            raise TypeError('Expected string or module, got %r' % package.__class__.__name__)
        self._ensure_package(package)
        if is_ns:
            return _AnsibleNSTraversable(*package.submodule_search_locations)
        return pathlib.Path(self._get_path(package)).parent