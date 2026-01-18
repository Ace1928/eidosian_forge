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
class ControllerNotSupportedError(ApplicationError):
    """Option(s) were specified which do not provide support for the controller and would be ignored because they are irrelevant for the target."""

    def __init__(self, context: str) -> None:
        super().__init__(f'Environment `{context}` does not provide a Python version supported by the controller.')