from __future__ import annotations
import abc
import shlex
from .util import (
class Doas(Become):
    """Become using 'doas'."""

    @property
    def method(self) -> str:
        """The name of the Ansible become plugin that is equivalent to this."""
        raise NotImplementedError('Ansible has no built-in doas become plugin.')

    def prepare_command(self, command: list[str]) -> list[str]:
        """Return the given command, if any, with privilege escalation."""
        become = ['doas', '-n']
        if command:
            become.extend(['sh', '-c', shlex.join(command)])
        else:
            become.extend(['-s'])
        return become