from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
class ShellScriptTemplate:
    """A simple substitution template for shell scripts."""

    def __init__(self, template: str) -> None:
        self.template = template

    def substitute(self, **kwargs: t.Union[str, list[str]]) -> str:
        """Return a string templated with the given arguments."""
        kvp = dict(((k, self.quote(v)) for k, v in kwargs.items()))
        pattern = re.compile('#{(?P<name>[^}]+)}')
        value = pattern.sub(lambda match: kvp[match.group('name')], self.template)
        return value

    @staticmethod
    def quote(value: t.Union[str, list[str]]) -> str:
        """Return a shell quoted version of the given value."""
        if isinstance(value, list):
            return shlex.quote(' '.join(value))
        return shlex.quote(value)