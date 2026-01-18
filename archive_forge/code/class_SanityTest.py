from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
class SanityTest(metaclass=abc.ABCMeta):
    """Sanity test base class."""
    ansible_only = False

    def __init__(self, name: t.Optional[str]=None) -> None:
        if not name:
            name = self.__class__.__name__
            name = re.sub('Test$', '', name)
            name = re.sub('(.)([A-Z][a-z]+)', '\\1-\\2', name).lower()
        self.name = name
        self.enabled = True
        self.optional_error_codes: set[str] = set()

    @property
    def error_code(self) -> t.Optional[str]:
        """Error code for ansible-test matching the format used by the underlying test program, or None if the program does not use error codes."""
        return None

    @property
    def can_ignore(self) -> bool:
        """True if the test supports ignore entries."""
        return True

    @property
    def can_skip(self) -> bool:
        """True if the test supports skip entries."""
        return not self.all_targets and (not self.no_targets)

    @property
    def all_targets(self) -> bool:
        """True if test targets will not be filtered using includes, excludes, requires or changes. Mutually exclusive with no_targets."""
        return False

    @property
    def no_targets(self) -> bool:
        """True if the test does not use test targets. Mutually exclusive with all_targets."""
        return False

    @property
    def include_directories(self) -> bool:
        """True if the test targets should include directories."""
        return False

    @property
    def include_symlinks(self) -> bool:
        """True if the test targets should include symlinks."""
        return False

    @property
    def py2_compat(self) -> bool:
        """True if the test only applies to code that runs on Python 2.x."""
        return False

    @property
    def supported_python_versions(self) -> t.Optional[tuple[str, ...]]:
        """A tuple of supported Python versions or None if the test does not depend on specific Python versions."""
        return CONTROLLER_PYTHON_VERSIONS

    def origin_hook(self, args: SanityConfig) -> None:
        """This method is called on the origin, before the test runs or delegation occurs."""

    def filter_targets(self, targets: list[TestTarget]) -> list[TestTarget]:
        """Return the given list of test targets, filtered to include only those relevant for the test."""
        if self.no_targets:
            return []
        raise NotImplementedError('Sanity test "%s" must implement "filter_targets" or set "no_targets" to True.' % self.name)

    def filter_targets_by_version(self, args: SanityConfig, targets: list[TestTarget], python_version: str) -> list[TestTarget]:
        """Return the given list of test targets, filtered to include only those relevant for the test, taking into account the Python version."""
        del python_version
        targets = self.filter_targets(targets)
        if self.py2_compat:
            content_config = get_content_config(args)
            if content_config.py2_support:
                targets = self.filter_remote_targets(targets)
            else:
                targets = []
        return targets

    @staticmethod
    def filter_remote_targets(targets: list[TestTarget]) -> list[TestTarget]:
        """Return a filtered list of the given targets, including only those that require support for remote-only Python versions."""
        targets = [target for target in targets if is_subdir(target.path, data_context().content.module_path) or is_subdir(target.path, data_context().content.module_utils_path) or is_subdir(target.path, data_context().content.unit_module_path) or is_subdir(target.path, data_context().content.unit_module_utils_path) or re.search('^%s/.*/library/' % re.escape(data_context().content.integration_targets_path), target.path) or (data_context().content.is_ansible and (is_subdir(target.path, 'test/lib/ansible_test/_util/target/') or re.search('^test/support/integration/.*/(modules|module_utils)/', target.path) or re.search('^lib/ansible/utils/collection_loader/', target.path)))]
        return targets