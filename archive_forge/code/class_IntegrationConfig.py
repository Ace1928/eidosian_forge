from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
class IntegrationConfig(TestConfig):
    """Configuration for the integration command."""

    def __init__(self, args: t.Any, command: str) -> None:
        super().__init__(args, command)
        self.start_at: str = args.start_at
        self.start_at_task: str = args.start_at_task
        self.allow_destructive: bool = args.allow_destructive
        self.allow_root: bool = args.allow_root
        self.allow_disabled: bool = args.allow_disabled
        self.allow_unstable: bool = args.allow_unstable
        self.allow_unstable_changed: bool = args.allow_unstable_changed
        self.allow_unsupported: bool = args.allow_unsupported
        self.retry_on_error: bool = args.retry_on_error
        self.continue_on_error: bool = args.continue_on_error
        self.debug_strategy: bool = args.debug_strategy
        self.changed_all_target: str = args.changed_all_target
        self.changed_all_mode: str = args.changed_all_mode
        self.list_targets: bool = args.list_targets
        self.tags = args.tags
        self.skip_tags = args.skip_tags
        self.diff = args.diff
        self.no_temp_workdir: bool = args.no_temp_workdir
        self.no_temp_unicode: bool = args.no_temp_unicode
        if self.list_targets:
            self.explain = True
            self.display_stderr = True

    def get_ansible_config(self) -> str:
        """Return the path to the Ansible config for the given config."""
        ansible_config_relative_path = os.path.join(data_context().content.integration_path, '%s.cfg' % self.command)
        ansible_config_path = os.path.join(data_context().content.root, ansible_config_relative_path)
        if not os.path.exists(ansible_config_path):
            ansible_config_path = super().get_ansible_config()
        return ansible_config_path