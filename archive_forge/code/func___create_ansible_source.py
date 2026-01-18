from __future__ import annotations
import collections.abc as c
import dataclasses
import os
import typing as t
from .util import (
from .provider import (
from .provider.source import (
from .provider.source.unversioned import (
from .provider.source.installed import (
from .provider.source.unsupported import (
from .provider.layout import (
from .provider.layout.unsupported import (
def __create_ansible_source(self):
    """Return a tuple of Ansible source files with both absolute and relative paths."""
    if not ANSIBLE_SOURCE_ROOT:
        sources = []
        source_provider = InstalledSource(ANSIBLE_LIB_ROOT)
        sources.extend(((os.path.join(source_provider.root, path), os.path.join('lib', 'ansible', path)) for path in source_provider.get_paths(source_provider.root)))
        source_provider = InstalledSource(ANSIBLE_TEST_ROOT)
        sources.extend(((os.path.join(source_provider.root, path), os.path.join('test', 'lib', 'ansible_test', path)) for path in source_provider.get_paths(source_provider.root)))
        return tuple(sources)
    if self.content.is_ansible:
        return tuple(((os.path.join(self.content.root, path), path) for path in self.content.all_files()))
    try:
        source_provider = find_path_provider(SourceProvider, self.__source_providers, ANSIBLE_SOURCE_ROOT, False)
    except ProviderNotFoundForPath:
        source_provider = UnversionedSource(ANSIBLE_SOURCE_ROOT)
    return tuple(((os.path.join(source_provider.root, path), path) for path in source_provider.get_paths(source_provider.root)))