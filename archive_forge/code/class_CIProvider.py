from __future__ import annotations
import abc
import base64
import json
import os
import tempfile
import typing as t
from ..encoding import (
from ..io import (
from ..config import (
from ..util import (
class CIProvider(metaclass=abc.ABCMeta):
    """Base class for CI provider plugins."""
    priority = 500

    @staticmethod
    @abc.abstractmethod
    def is_supported() -> bool:
        """Return True if this provider is supported in the current running environment."""

    @property
    @abc.abstractmethod
    def code(self) -> str:
        """Return a unique code representing this provider."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return descriptive name for this provider."""

    @abc.abstractmethod
    def generate_resource_prefix(self) -> str:
        """Return a resource prefix specific to this CI provider."""

    @abc.abstractmethod
    def get_base_commit(self, args: CommonConfig) -> str:
        """Return the base commit or an empty string."""

    @abc.abstractmethod
    def detect_changes(self, args: TestConfig) -> t.Optional[list[str]]:
        """Initialize change detection."""

    @abc.abstractmethod
    def supports_core_ci_auth(self) -> bool:
        """Return True if Ansible Core CI is supported."""

    @abc.abstractmethod
    def prepare_core_ci_auth(self) -> dict[str, t.Any]:
        """Return authentication details for Ansible Core CI."""

    @abc.abstractmethod
    def get_git_details(self, args: CommonConfig) -> t.Optional[dict[str, t.Any]]:
        """Return details about git in the current environment."""