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
class DockerCompletionConfig(PythonCompletionConfig):
    """Configuration for Docker containers."""
    image: str = ''
    seccomp: str = 'default'
    cgroup: str = CGroupVersion.V1_V2.value
    audit: str = AuditMode.REQUIRED.value
    placeholder: bool = False

    @property
    def is_default(self) -> bool:
        """True if the completion entry is only used for defaults, otherwise False."""
        return False

    @property
    def audit_enum(self) -> AuditMode:
        """The audit requirements for the container. Raises an exception if the value is invalid."""
        try:
            return AuditMode(self.audit)
        except ValueError:
            raise ValueError(f'Docker completion entry "{self.name}" has an invalid value "{self.audit}" for the "audit" setting.') from None

    @property
    def cgroup_enum(self) -> CGroupVersion:
        """The control group version(s) required by the container. Raises an exception if the value is invalid."""
        try:
            return CGroupVersion(self.cgroup)
        except ValueError:
            raise ValueError(f'Docker completion entry "{self.name}" has an invalid value "{self.cgroup}" for the "cgroup" setting.') from None

    def __post_init__(self):
        if not self.image:
            raise Exception(f'Docker completion entry "{self.name}" must provide an "image" setting.')
        if not self.supported_pythons and (not self.placeholder):
            raise Exception(f'Docker completion entry "{self.name}" must provide a "python" setting.')
        assert self.audit_enum
        assert self.cgroup_enum