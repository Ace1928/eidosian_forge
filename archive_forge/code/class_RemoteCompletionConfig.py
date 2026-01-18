from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
@dataclasses.dataclass(frozen=True)
class RemoteCompletionConfig(CompletionConfig):
    """Base class for completion configuration of remote environments provisioned through Ansible Core CI."""
    provider: t.Optional[str] = None
    arch: t.Optional[str] = None

    @property
    def platform(self) -> str:
        """The name of the platform."""
        return self.name.partition('/')[0]

    @property
    def version(self) -> str:
        """The version of the platform."""
        return self.name.partition('/')[2]

    @property
    def is_default(self) -> bool:
        """True if the completion entry is only used for defaults, otherwise False."""
        return not self.version

    def __post_init__(self):
        if not self.provider:
            raise Exception(f'Remote completion entry "{self.name}" must provide a "provider" setting.')
        if not self.arch:
            raise Exception(f'Remote completion entry "{self.name}" must provide a "arch" setting.')