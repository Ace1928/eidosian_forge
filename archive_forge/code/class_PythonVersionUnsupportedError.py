from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
class PythonVersionUnsupportedError(ApplicationError):
    """A Python version was requested for a context which does not support that version."""

    def __init__(self, context: str, version: str, versions: c.Iterable[str]) -> None:
        super().__init__(f'Python {version} is not supported by environment `{context}`. Supported Python version(s) are: {', '.join(versions)}')