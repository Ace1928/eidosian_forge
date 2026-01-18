from __future__ import annotations
import abc
import dataclasses
import enum
import os
import pickle
import sys
import typing as t
from .constants import (
from .io import (
from .completion import (
from .util import (
@dataclasses.dataclass
class PosixRemoteConfig(RemoteConfig, ControllerHostConfig, PosixConfig):
    """Configuration for a POSIX remote host."""
    become: t.Optional[str] = None

    def get_defaults(self, context: HostContext) -> PosixRemoteCompletionConfig:
        """Return the default settings."""
        return filter_completion(remote_completion()).get(self.name) or remote_completion().get(self.platform) or PosixRemoteCompletionConfig(name=self.name, placeholder=True)

    def get_default_targets(self, context: HostContext) -> list[ControllerConfig]:
        """Return the default targets for this host config."""
        if self.name in filter_completion(remote_completion()):
            defaults = self.get_defaults(context)
            pythons = {version: defaults.get_python_path(version) for version in defaults.supported_pythons}
        else:
            pythons = {context.controller_config.python.version: context.controller_config.python.path}
        return [ControllerConfig(python=NativePythonConfig(version=version, path=path)) for version, path in pythons.items()]

    def apply_defaults(self, context: HostContext, defaults: CompletionConfig) -> None:
        """Apply default settings."""
        assert isinstance(defaults, PosixRemoteCompletionConfig)
        super().apply_defaults(context, defaults)
        self.become = self.become or defaults.become

    @property
    def have_root(self) -> bool:
        """True if root is available, otherwise False."""
        return True