from __future__ import annotations
import typing as t
from .util import (
from .io import (
from .diff import (
class ChangeDescription:
    """Description of changes."""

    def __init__(self) -> None:
        self.command: str = ''
        self.changed_paths: list[str] = []
        self.deleted_paths: list[str] = []
        self.regular_command_targets: dict[str, list[str]] = {}
        self.focused_command_targets: dict[str, list[str]] = {}
        self.no_integration_paths: list[str] = []

    @property
    def targets(self) -> t.Optional[list[str]]:
        """Optional list of target names."""
        return self.regular_command_targets.get(self.command)

    @property
    def focused_targets(self) -> t.Optional[list[str]]:
        """Optional list of focused target names."""
        return self.focused_command_targets.get(self.command)

    def to_dict(self) -> dict[str, t.Any]:
        """Return a dictionary representation of the change description."""
        return dict(command=self.command, changed_paths=self.changed_paths, deleted_paths=self.deleted_paths, regular_command_targets=self.regular_command_targets, focused_command_targets=self.focused_command_targets, no_integration_paths=self.no_integration_paths)

    @staticmethod
    def from_dict(data: dict[str, t.Any]) -> ChangeDescription:
        """Return a change description loaded from the given dictionary."""
        changes = ChangeDescription()
        changes.command = data['command']
        changes.changed_paths = data['changed_paths']
        changes.deleted_paths = data['deleted_paths']
        changes.regular_command_targets = data['regular_command_targets']
        changes.focused_command_targets = data['focused_command_targets']
        changes.no_integration_paths = data['no_integration_paths']
        return changes