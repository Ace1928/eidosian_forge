from __future__ import annotations
import abc
import shlex
from .util import (
class Become(metaclass=abc.ABCMeta):
    """Base class for become implementations."""

    @classmethod
    def name(cls) -> str:
        """The name of this plugin."""
        return cls.__name__.lower()

    @property
    @abc.abstractmethod
    def method(self) -> str:
        """The name of the Ansible become plugin that is equivalent to this."""

    @abc.abstractmethod
    def prepare_command(self, command: list[str]) -> list[str]:
        """Return the given command, if any, with privilege escalation."""