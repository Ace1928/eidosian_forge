from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
class PipUnavailableError(ApplicationError):
    """Exception raised when pip is not available."""

    def __init__(self, python: PythonConfig) -> None:
        super().__init__(f'Python {python.version} at "{python.path}" does not have pip available.')