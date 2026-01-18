from __future__ import annotations
import abc
import collections
import os
import typing as t
from ...util import (
from .. import (
class ContentLayout(Layout):
    """Information about the current Ansible content being tested."""

    def __init__(self, root: str, paths: list[str], plugin_paths: dict[str, str], collection: t.Optional[CollectionDetail], test_path: str, results_path: str, sanity_path: str, sanity_messages: t.Optional[LayoutMessages], integration_path: str, integration_targets_path: str, integration_vars_path: str, integration_messages: t.Optional[LayoutMessages], unit_path: str, unit_module_path: str, unit_module_utils_path: str, unit_messages: t.Optional[LayoutMessages], unsupported: bool | list[str]=False) -> None:
        super().__init__(root, paths)
        self.plugin_paths = plugin_paths
        self.collection = collection
        self.test_path = test_path
        self.results_path = results_path
        self.sanity_path = sanity_path
        self.sanity_messages = sanity_messages
        self.integration_path = integration_path
        self.integration_targets_path = integration_targets_path
        self.integration_vars_path = integration_vars_path
        self.integration_messages = integration_messages
        self.unit_path = unit_path
        self.unit_module_path = unit_module_path
        self.unit_module_utils_path = unit_module_utils_path
        self.unit_messages = unit_messages
        self.unsupported = unsupported
        self.is_ansible = root == ANSIBLE_SOURCE_ROOT

    @property
    def prefix(self) -> str:
        """Return the collection prefix or an empty string if not a collection."""
        if self.collection:
            return self.collection.prefix
        return ''

    @property
    def module_path(self) -> t.Optional[str]:
        """Return the path where modules are found, if any."""
        return self.plugin_paths.get('modules')

    @property
    def module_utils_path(self) -> t.Optional[str]:
        """Return the path where module_utils are found, if any."""
        return self.plugin_paths.get('module_utils')

    @property
    def module_utils_powershell_path(self) -> t.Optional[str]:
        """Return the path where powershell module_utils are found, if any."""
        if self.is_ansible:
            return os.path.join(self.plugin_paths['module_utils'], 'powershell')
        return self.plugin_paths.get('module_utils')

    @property
    def module_utils_csharp_path(self) -> t.Optional[str]:
        """Return the path where csharp module_utils are found, if any."""
        if self.is_ansible:
            return os.path.join(self.plugin_paths['module_utils'], 'csharp')
        return self.plugin_paths.get('module_utils')