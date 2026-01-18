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
@dataclasses.dataclass(frozen=True)
class HostSettings:
    """Host settings for the controller and targets."""
    controller: ControllerHostConfig
    targets: list[HostConfig]
    skipped_python_versions: list[str]
    filtered_args: list[str]
    controller_fallback: t.Optional[FallbackDetail]

    def serialize(self, path: str) -> None:
        """Serialize the host settings to the given path."""
        with open_binary_file(path, 'wb') as settings_file:
            pickle.dump(self, settings_file)

    @staticmethod
    def deserialize(path: str) -> HostSettings:
        """Deserialize host settings from the path."""
        with open_binary_file(path) as settings_file:
            return pickle.load(settings_file)

    def apply_defaults(self) -> None:
        """Apply defaults to the host settings."""
        context = HostContext(controller_config=None)
        self.controller.apply_defaults(context, self.controller.get_defaults(context))
        for target in self.targets:
            context = HostContext(controller_config=self.controller)
            target.apply_defaults(context, target.get_defaults(context))